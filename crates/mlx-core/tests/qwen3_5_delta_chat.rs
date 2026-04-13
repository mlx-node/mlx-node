//! Gated integration test for the session-based chat delta path.
//!
//! This test exercises the Phase 2 production surface — `chat_session_start`
//! for turn 1 and `chat_session_continue` for turns 2..=4 — and validates
//! that TTFT stays roughly flat across turns. That is direct evidence the
//! KV caches are being reused and each new turn only pays for its delta
//! prefill, not a full re-prefill of the accumulating history.
//!
//! The test is gated because it needs a real Qwen3.5 Dense checkpoint on
//! disk. Run it manually with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/qwen3.5-0.8b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_5_delta_chat -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_MODEL_PATH` the test early-returns and passes
//! trivially so it still compiles as part of `cargo test`.

use std::path::Path;
use std::time::Instant;

use mlx_core::models::qwen3_5::model::{ChatConfig, Qwen3_5Model};
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
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn session_path_keeps_ttft_flat_across_turns() {
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

    // --- Turn 1: chat_session_start establishes a clean session ---
    //
    // Unlike the legacy `chat()` path, this uses `<|im_end|>` as eos so the
    // cached history ends on a clean ChatML boundary that the subsequent
    // `chat_session_continue` deltas can append to.
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
    //
    // The session state is owned entirely by the model thread — the
    // caller just passes plain user strings. `chat_session_continue_sync`
    // builds the ChatML delta, tokenizes it, and prefills on top of the
    // live caches. No template rendering, no prefix matching.
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

    // 2. TTFT stays flat (<=1.5x of turn 1) across all turns. The broken
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
    //    session-start path happens to do — comparing turn 4 to turn 2
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

// ---------------------------------------------------------------------
// Streaming session path tests
// ---------------------------------------------------------------------
//
// Cover the Phase-3 streaming surface:
//   - `chat_stream_session_start_for_test` — turn 1, resets caches, uses
//     `<|im_end|>` as eos (matches `chat_session_start`).
//   - `chat_stream_session_continue_for_test` — turns 2..N, prefill the
//     ChatML delta on top of the live caches, stream the reply token-by-
//     token through the supplied mpsc receiver.
//
// We use the `*_for_test` helpers exported from `Qwen3_5Model` rather
// than `chat_stream_session_*` directly because the latter require a JS
// `ThreadsafeFunction` callback, which we can't construct outside a
// NAPI host. The `_for_test` variants expose the underlying mpsc
// receiver directly so a pure-Rust integration test can iterate it.
//
// Gated the same way as the non-streaming test above — unset
// `MLX_TEST_MODEL_PATH` → early return → trivial pass.

/// Drain the streaming receiver for one turn into a (chunks, ttft_ms)
/// snapshot. TTFT is measured from call time to the first chunk (delta
/// or final). The final `done: true` chunk must be observed or the
/// helper panics — a partial stream would break the delta-cache
/// invariants expected by subsequent turns.
async fn drain_stream_turn(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<
        napi::Result<mlx_core::models::qwen3_5::model::ChatStreamChunk>,
    >,
) -> (
    Vec<mlx_core::models::qwen3_5::model::ChatStreamChunk>,
    f64,
    bool,
) {
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
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn stream_session_path_keeps_ttft_flat_across_turns() {
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

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");

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

    // 1. prompt_tokens grows across turns
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

    // 2. TTFT stays flat (<=1.5x of turn 1)
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

    // 3. Turn 4 should be in the same flat-TTFT regime as turn 2
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
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn stream_session_cancellation_preserves_cache_for_next_turn() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };

    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");

    // Turn 1: run a normal session-start stream to prime the cache.
    // Use 128 tokens so cancellation has room to hit mid-stream.
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
    // Cancellation should still produce a final done chunk before the
    // channel closes — the decode loop sets finish_reason="cancelled"
    // and falls through to the final-chunk emission.
    assert!(
        saw_done,
        "cancellation didn't produce a final done chunk (collected={collected})",
    );
    // Race-tolerant assertion: the decode loop may naturally reach
    // `stop`/`length` on the very next step after `handle.cancel()`
    // is called, before the cancellation flag is checked. Accept any
    // non-None finish_reason here — the cache-consistency follow-up
    // continue below is the real invariant check.
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
    // the partial generated tokens so this MUST succeed — the session
    // is still consistent, just with a partial previous reply.
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
// Step-1 coverage: image-in-continue rejection + tool-result round-trip
// ---------------------------------------------------------------------
//
// These assertions guard the new `images` parameter on
// `chat_session_continue` (must reject non-empty with a typed error
// prefixed by `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`) and the new
// `chat_session_continue_tool` entry point (must round-trip a
// tool-response delta through the session path and return a non-empty
// reply). Both are gated on `MLX_TEST_MODEL_PATH` the same way as the
// tests above — no model, trivial pass.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn session_continue_rejects_images_with_restart_prefix() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");

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
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn session_continue_tool_round_trips() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");

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

/// Gated VLM session-start smoke test: verifies that
/// `chat_session_start` now accepts image messages and produces a reply.
///
/// Skipped unless `MLX_TEST_VLM_MODEL_PATH` AND
/// `MLX_TEST_VLM_IMAGE_PATH` are set — the full VLM session coverage
/// is deferred to a later step; this test exists only to lock in the
/// fact that the text-only guard was actually removed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_VLM_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH for a Qwen3.5 VLM checkpoint + test image"]
async fn session_start_accepts_images_for_vlm() {
    let Ok(model_path) = std::env::var("MLX_TEST_VLM_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_VLM_MODEL_PATH unset");
        return;
    };
    let Ok(image_path) = std::env::var("MLX_TEST_VLM_IMAGE_PATH") else {
        eprintln!("skipping: MLX_TEST_VLM_IMAGE_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_VLM_MODEL_PATH does not exist: {}",
        model_path
    );
    let img_file = Path::new(&image_path);
    assert!(
        img_file.exists(),
        "MLX_TEST_VLM_IMAGE_PATH does not exist: {}",
        image_path
    );

    let image_bytes = std::fs::read(&image_path).expect("failed to read image file");
    let image_uint8: napi::bindgen_prelude::Uint8Array =
        napi::bindgen_prelude::Uint8Array::new(image_bytes);

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 VLM model");

    let message = ChatMessage {
        role: "user".to_string(),
        content: "Describe this image briefly.".to_string(),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
        images: Some(vec![image_uint8]),
    };

    let cfg = chat_config_default(32);
    let result = model
        .chat_session_start(vec![message], Some(cfg))
        .await
        .expect("chat_session_start with VLM image failed");

    assert!(
        result.finish_reason == "stop" || result.finish_reason == "length",
        "unexpected finish_reason: {}",
        result.finish_reason
    );
    assert!(
        result.num_tokens > 0,
        "VLM session-start returned zero generated tokens: {:?}",
        result
    );
}
