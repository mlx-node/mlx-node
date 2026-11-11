//! Gated integration test for the session-based chat delta path.
//!
//! This test exercises the production surface — `chat_session_start`
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

use mlx_core::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
use mlx_core::tokenizer::{ChatMessage, ToolCall};

fn chat_config_default(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        cache_salt: None,
        cache_owner_id: None,
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
        reasoning_effort: None,
        thinking_token_budget: Some(32), // keep it quick
        include_reasoning: Some(true),
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: None,
        mtp_depth: None,
        mtp_adaptive_depth: None,
    }
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

#[derive(Clone)]
struct AssistantTurn {
    text: String,
    tool_calls: Option<Vec<ToolCall>>,
    thinking: Option<String>,
    thinking_enabled: bool,
}

fn replay_tool_calls(tool_calls: &[mlx_core::tools::ToolCallResult]) -> Option<Vec<ToolCall>> {
    (!tool_calls.is_empty()).then(|| {
        tool_calls
            .iter()
            .map(|call| ToolCall {
                id: Some(call.id.clone()),
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            })
            .collect()
    })
}

fn assistant_turn_from_result(result: &ChatResult) -> AssistantTurn {
    AssistantTurn {
        text: result.text.clone(),
        tool_calls: replay_tool_calls(&result.tool_calls),
        thinking: result.thinking.clone(),
        thinking_enabled: result.thinking_enabled,
    }
}

fn assistant_turn_from_chunk(chunk: &ChatStreamChunk) -> AssistantTurn {
    assert!(chunk.done, "assistant replay requires a terminal chunk");
    AssistantTurn {
        text: chunk.text.clone(),
        tool_calls: replay_tool_calls(chunk.tool_calls.as_deref().unwrap_or_default()),
        thinking: chunk.thinking.clone(),
        thinking_enabled: chunk
            .thinking_enabled
            .expect("terminal stream chunk missing thinking_enabled provenance"),
    }
}

fn assistant_message(turn: &AssistantTurn) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: turn.text.clone(),
        tool_calls: turn.tool_calls.clone(),
        tool_call_id: None,
        is_error: None,
        reasoning_content: turn.thinking.clone(),
        thinking_enabled: Some(turn.thinking_enabled),
        images: None,
        audio: None,
    }
}

fn transcript_with_next_user(
    user_turns: &[String],
    assistant_turns: &[AssistantTurn],
    next_user: &str,
) -> Vec<ChatMessage> {
    assert_eq!(
        user_turns.len(),
        assistant_turns.len(),
        "every historical user turn must have an assistant reply",
    );
    let mut messages = Vec::with_capacity(user_turns.len() * 2 + 1);
    for (user, assistant) in user_turns.iter().zip(assistant_turns) {
        messages.push(user_message(user));
        messages.push(assistant_message(assistant));
    }
    messages.push(user_message(next_user));
    messages
}

fn tool_message(tool_call_id: &str, content: &str, is_error: Option<bool>) -> ChatMessage {
    ChatMessage {
        role: "tool".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: Some(tool_call_id.to_string()),
        is_error,
        reasoning_content: None,
        thinking_enabled: None,
        images: None,
        audio: None,
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
    let model = Qwen3_5Model::load(model_path.clone(), None)
        .await
        .expect("failed to load Qwen3.5 model");

    /// Compact per-turn snapshot used for structural assertions below.
    #[derive(Debug, Clone)]
    struct TurnSnapshot {
        ttft_ms: f64,
        prompt_tokens: u32,
        cached_tokens: u32,
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
        cached_tokens: r1.cached_tokens,
    };
    println!(
        "turn 1 ttft={:.1}ms prompt_tokens={} num_tokens={}",
        turn1.ttft_ms, turn1.prompt_tokens, r1.num_tokens
    );

    // --- Turns 2..=4: chat_session_continue (delta path) ---
    //
    // The caller replays the complete structured history. The model-provided
    // template renders it, and the session engine reuses the exact cached token
    // prefix while prefilling only the newly rendered suffix.
    let user_followups = [
        "And in another word?",
        "Any synonym?",
        "One more, different?",
    ];
    let mut snapshots: Vec<TurnSnapshot> = vec![turn1.clone()];
    let mut user_turns = vec!["Say hi in one short word.".to_string()];
    let mut assistant_turns = vec![assistant_turn_from_result(&r1)];

    for (idx, next_user) in user_followups.iter().enumerate() {
        let turn_idx = idx + 2;
        let cfg = chat_config_default(64);
        let result = model
            .chat_session_continue(
                transcript_with_next_user(&user_turns, &assistant_turns, next_user),
                Some(cfg),
            )
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
            cached_tokens: result.cached_tokens,
        });

        assert!(
            result.finish_reason == "stop" || result.finish_reason == "length",
            "unexpected finish_reason: {}",
            result.finish_reason
        );
        user_turns.push((*next_user).to_string());
        assistant_turns.push(assistant_turn_from_result(&result));
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

    // 2. The delta path reuses the entire prior context (the live KV cache)
    //    and freshly prefills ONLY the new turn's ChatML delta. We assert on
    //    that token accounting rather than wall-clock TTFT: TTFT flakes on
    //    shared CI runners (tiny warm-GPU times dominated by fixed per-turn
    //    setup, not prefill work) and is a WEAK signal — a path that silently
    //    rebuilt the cache each turn could still post a fast TTFT on a fast
    //    machine. `cached_tokens` is the deterministic proof.

    // 2a. cached_tokens GROWS across delta turns: each delta reuses the full,
    //     ever-longer prior context. A regressed delta that desynced and
    //     re-prefilled from scratch reports cached_tokens == 0 (the engine's
    //     `is_delta && !desynced` gate), so non-zero, strictly-growing
    //     cached_tokens is direct evidence the cache is reused, not rebuilt.
    assert_eq!(
        turn1.cached_tokens, 0,
        "turn 1 cold-starts but reported cached_tokens={}",
        turn1.cached_tokens
    );
    assert!(
        turn2.cached_tokens > 0,
        "delta turn 2 reused nothing (cached_tokens=0) — cache was rebuilt. snapshots: {:?}",
        snapshots
    );
    assert!(
        turn3.cached_tokens > turn2.cached_tokens,
        "delta turn 3 didn't grow its reused prefix ({} -> {}). snapshots: {:?}",
        turn2.cached_tokens,
        turn3.cached_tokens,
        snapshots
    );
    assert!(
        turn4.cached_tokens > turn3.cached_tokens,
        "delta turn 4 didn't grow its reused prefix ({} -> {}). snapshots: {:?}",
        turn3.cached_tokens,
        turn4.cached_tokens,
        snapshots
    );

    // 2b. The freshly-prefilled span (prompt_tokens - cached_tokens) stays
    //     small and FLAT even as the context grows — it is just one new user
    //     turn's ChatML each time, never the whole transcript. A full
    //     re-prefill regression would make turn 4's fresh span scale with the
    //     accumulated history instead.
    let uncached = |t: &TurnSnapshot| t.prompt_tokens.saturating_sub(t.cached_tokens);
    assert!(
        uncached(turn4) < turn4.cached_tokens,
        "delta turn 4 prefilled more than it reused — work is not flat: \
         prompt={} cached={} uncached={}. snapshots: {:?}",
        turn4.prompt_tokens,
        turn4.cached_tokens,
        uncached(turn4),
        snapshots
    );
    assert!(
        uncached(turn4) <= uncached(turn2) * 2,
        "delta turn 4's fresh prefill ({}) ballooned vs turn 2's ({}) — TTFT \
         would regress. snapshots: {:?}",
        uncached(turn4),
        uncached(turn2),
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
    mut rx: tokio::sync::mpsc::Receiver<napi::Result<mlx_core::engine::types::ChatStreamChunk>>,
) -> (Vec<mlx_core::engine::types::ChatStreamChunk>, f64, bool) {
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

    let model = Qwen3_5Model::load(model_path.clone(), None)
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
    let mut user_turns = vec!["Say hi in one short word.".to_string()];
    let mut assistant_turns = vec![assistant_turn_from_chunk(final1)];

    for (idx, next_user) in user_followups.iter().enumerate() {
        let turn_idx = idx + 2;
        let cfg = chat_config_default(64);
        let (_handle, rx) = model
            .chat_stream_session_continue_for_test(
                transcript_with_next_user(&user_turns, &assistant_turns, next_user),
                Some(cfg),
            )
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
        user_turns.push((*next_user).to_string());
        assistant_turns.push(assistant_turn_from_chunk(last));
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

    let model = Qwen3_5Model::load(model_path.clone(), None)
        .await
        .expect("failed to load Qwen3.5 model");

    // Turn 1: run a normal session-start stream to prime the cache.
    // Use 128 tokens so cancellation has room to hit mid-stream.
    let turn1_cfg = ChatConfig {
        cache_salt: None,
        cache_owner_id: None,
        cache_root_owner_id: None,
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
    let mut terminal1: Option<ChatStreamChunk> = None;
    while let Some(result) = rx1.recv().await {
        let chunk = result.expect("stream error during cancel test");
        if chunk.done {
            saw_done = true;
            finish_reason = chunk.finish_reason.clone();
            terminal1 = Some(chunk);
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
    let terminal1 = terminal1.expect("cancelled turn missing terminal chunk");
    let (_handle2, rx2) = model
        .chat_stream_session_continue_for_test(
            transcript_with_next_user(
                &["Count slowly to twenty.".to_string()],
                &[assistant_turn_from_chunk(&terminal1)],
                "What number were you on?",
            ),
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

    let model = Qwen3_5Model::load(model_path.clone(), None)
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
    let mut invalid_media_message = user_message("What now?");
    invalid_media_message.images = Some(vec![dummy_image]);

    let cfg = chat_config_default(32);
    let err = model
        .chat_session_continue(vec![invalid_media_message], Some(cfg))
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

    let model = Qwen3_5Model::load(model_path.clone(), None)
        .await
        .expect("failed to load Qwen3.5 model");

    // Start a session so the tool-result delta has a live cache to
    // prefill on top of.
    let turn1_cfg = chat_config_default(32);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let first = model
        .chat_session_start(turn1_messages, Some(turn1_cfg))
        .await
        .expect("turn 1 chat_session_start failed");

    let tool_cfg = chat_config_default(32);
    let result = model
        .chat_session_continue_tool(
            vec![
                user_message("Say hi in one short word."),
                assistant_message(&assistant_turn_from_result(&first)),
                tool_message("dummy_id", "result content", None),
            ],
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

    let model = Qwen3_5Model::load(model_path.clone(), None)
        .await
        .expect("failed to load Qwen3.5 VLM model");

    let message = ChatMessage {
        role: "user".to_string(),
        content: "Describe this image briefly.".to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: Some(vec![image_uint8]),
        audio: None,
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
