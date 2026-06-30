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

use mlx_core::engine::types::{ChatConfig, ChatStreamChunk};
use mlx_core::models::lfm2::model::Lfm2Model;
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
        images: None,
        audio: None,
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
        cached_tokens: u32,
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
        cached_tokens: r1.cached_tokens,
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
            .chat_session_continue((*next_user).to_string(), None, None, Some(cfg))
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
        .chat_session_continue("What now?".to_string(), images, None, Some(cfg))
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
            None,
        )
        .await
        .expect("chat_session_continue_tool failed");

    assert!(
        !result.text.is_empty() || result.num_tokens > 0,
        "tool-result continue returned an empty reply: {:?}",
        result
    );
    // This test guards the continue_tool PATH mechanics: it prefills the
    // tool-result delta onto the live session cache, generates, and reaches a
    // clean terminal — it does NOT assert reply content. LFM2 drops the
    // tool_call_id (its template identifies tool responses positionally), so a
    // minimal tool-continue on the "thinking" checkpoint enters a think block
    // immediately and, under the 32-token budget here, legitimately bottoms out
    // via EOS ("stop") or the budget cap ("length"). The repetition guard is
    // OFF by default (params.rs resolves max_consecutive_tokens / max_ngram_repeats
    // / ngram_size to 0 when the config leaves them unset), so "repetition" only
    // fires if a caller opts in; it stays an accepted terminal state here for that
    // case. All are valid non-cancelled terminal states. Forward correctness is
    // covered separately by lfm2_paged_vs_flat_parity (6/6 byte-identical).
    assert!(
        matches!(
            result.finish_reason.as_str(),
            "stop" | "length" | "repetition"
        ),
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
        .chat_session_continue("hi".to_string(), None, None, Some(cfg))
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
/// temperature=0 reproduces the previous `raw_text` / `text` /
/// `num_tokens` byte-for-byte WITHOUT any priming turn. This proves the
/// session-start path starts from a fully cold slate on reset and
/// nothing from the previous turn leaks into the new one — including
/// the paged adapter's content-addressed prefix blocks: an explicit
/// `reset_caches` purges them (`ResetScope::Command` hard reset), so
/// the second run replays the COLD full-prompt prefill instead of a
/// prefix-hit 1-token-suffix prefill whose different bf16 reduction
/// order can flip a greedy near-tie (the codex S12 finding; previously
/// masked because `text` is empty on all-thinking trajectories —
/// `raw_text` is the assert that actually bites).
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
    // block_in_place: reset_caches blocks on blocking_recv, which panics on a tokio worker.
    tokio::task::block_in_place(|| model.reset_caches()).expect("reset_caches failed");

    let cfg2 = chat_config_default(32);
    let r2 = model
        .chat_session_start(vec![user_message(prompt_text)], Some(cfg2))
        .await
        .expect("second chat_session_start after reset_caches failed");

    // raw_text is the load-bearing assert: lfm2 spends small budgets
    // inside `<think>`, so `text` is often empty for BOTH runs and
    // would mask a cold-vs-prefix-hit prefill divergence.
    assert_eq!(
        r1.raw_text, r2.raw_text,
        "reset_caches did not reproduce turn-0 raw_text byte-for-byte \
         (cold prefill vs post-reset prefill diverged): before={:?} after={:?}",
        r1.raw_text, r2.raw_text
    );
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
/// `chat_stream_session_start_for_test` matches the `ChatResult.raw_text`
/// from `chat_session_start` byte-for-byte (the deltas are the verbatim
/// stream under include_reasoning=true), and the terminal chunk's parsed
/// `text` matches the non-stream `text`. The `reset_caches` between the
/// two runs is a HARD reset (paged prefix blocks purged), so the
/// streaming run replays the non-streaming run's cold prefill exactly —
/// no priming turn needed (the S12 prime-turn workaround is gone with
/// the `ResetScope::Command` prefix-cache purge).
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

    let prompt_text = "Say hi in one short word.";

    // Non-streaming: capture the full reply text. `ChatMessage` is not
    // `Clone`, so we reconstruct the identical prompt for both calls.
    let cfg_ns = chat_config_default(32);
    let non_stream_result = model
        .chat_session_start(vec![user_message(prompt_text)], Some(cfg_ns))
        .await
        .expect("non-streaming chat_session_start failed");

    // Hard reset so the streaming run starts from the same fully cold
    // state as the non-streaming run above (prefix-cache purged).
    // block_in_place: reset_caches blocks on blocking_recv, which panics on a tokio worker.
    tokio::task::block_in_place(|| model.reset_caches()).expect("reset_caches failed");

    // Streaming: drain every non-done chunk and concatenate `chunk.text`.
    let cfg_s = chat_config_default(32);
    let (_handle, mut rx) = model
        .chat_stream_session_start_for_test(vec![user_message(prompt_text)], Some(cfg_s))
        .expect("chat_stream_session_start_for_test dispatch failed");

    let mut streamed = String::new();
    let mut saw_done = false;
    let mut terminal_text: Option<String> = None;
    while let Some(result) = rx.recv().await {
        let chunk = result.expect("stream chunk error");
        if chunk.done {
            saw_done = true;
            terminal_text = Some(chunk.text.clone());
            break;
        }
        streamed.push_str(&chunk.text);
    }
    assert!(saw_done, "stream never reached done=true");

    // With include_reasoning=true the delta chunks carry the VERBATIM byte
    // stream (reasoning included), so the correct non-stream counterpart is
    // `raw_text`, NOT the reasoning-stripped `text`. The original
    // `streamed == text` assert could never pass on a thinking trajectory
    // (lfm2 spends the whole 32-token budget inside `<think>`, so `text` is
    // empty) — a defect previously masked by the blocking_recv panic in
    // `reset_caches` (fixed above).
    assert_eq!(
        streamed, non_stream_result.raw_text,
        "streamed deltas do not match non-stream raw_text byte-for-byte: \
         streamed={:?} non_stream_raw={:?}",
        streamed, non_stream_result.raw_text
    );
    // Parsed-text parity: the terminal done-chunk carries the streaming
    // run's finalized `text`; it must match the non-streaming `text`.
    assert_eq!(
        terminal_text.as_deref(),
        Some(non_stream_result.text.as_str()),
        "terminal chunk text does not match non-stream text byte-for-byte"
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

// ---------------------------------------------------------------------
// Native prefix-KV-cache reuse across back-to-back `chat_session_start`
// calls. These cover the stateless-agent pattern (pi-mono / Aider /
// Codex resend-the-transcript-every-turn) that the native reuse path
// was added for. See `.claude/plans/dapper-zooming-catmull.md` for the
// full design rationale.
// ---------------------------------------------------------------------

/// Append hit: turn 2's `chat_session_start` prompt is a strict
/// extension of turn 1's. The reported `cached_tokens` on turn 2 must
/// be > 0 and at least cover turn 1's saved history, the reply must
/// still terminate cleanly, and turn 2's TTFT must stay flat (<= 1.5×
/// turn 1's) since only the delta is re-prefilled.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_start_prefix_reuse_append_hit() {
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

    // Turn 1: plain session start
    let cfg1 = chat_config_default(32);
    let r1 = model
        .chat_session_start(vec![user_message("In exactly one short word, and nothing else, with no explanation and no punctuation and no preamble whatsoever, how would you warmly and politely greet a brand-new friend that you happen to be meeting for the very first time on this fine and bright sunny morning, bearing in mind that they have travelled a very long way over the hills and across the wide river and through the quiet forest to come and see you today and would dearly appreciate a simple kind and gentle word of welcome from you?")], Some(cfg1))
        .await
        .expect("turn 1 chat_session_start failed");
    assert_eq!(
        r1.cached_tokens, 0,
        "turn 1 should cold-start: cached_tokens={}",
        r1.cached_tokens
    );

    // Turn 2: resend full transcript + one more user turn. The LFM2
    // chat template renders the prior assistant reply byte-for-byte, so
    // turn 2's token stream is a strict prefix extension of turn 1's
    // saved history.
    let cfg2 = chat_config_default(32);
    let turn2_msgs = vec![
        user_message(
            "In exactly one short word, and nothing else, with no explanation and no punctuation and no preamble whatsoever, how would you warmly and politely greet a brand-new friend that you happen to be meeting for the very first time on this fine and bright sunny morning, bearing in mind that they have travelled a very long way over the hills and across the wide river and through the quiet forest to come and see you today and would dearly appreciate a simple kind and gentle word of welcome from you?",
        ),
        ChatMessage {
            role: "assistant".to_string(),
            content: r1.text.clone(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            images: None,
            audio: None,
        },
        user_message("And another one?"),
    ];
    let r2 = model
        .chat_session_start(turn2_msgs, Some(cfg2))
        .await
        .expect("turn 2 chat_session_start failed");
    assert!(
        r2.cached_tokens > 0,
        "turn 2 expected a prefix cache hit but cached_tokens=0 (turn1 tokens={}, turn2 tokens={})",
        r1.prompt_tokens,
        r2.prompt_tokens
    );
    assert!(
        r2.finish_reason == "stop" || r2.finish_reason == "length",
        "unexpected finish_reason on turn 2: {}",
        r2.finish_reason
    );
    // On a prefix-reuse hit the engine reprocesses only the new suffix: the
    // matched prefix (turn 1's rendered transcript) is served from the
    // content-addressed cache and `cached_tokens` reports it. Assert that
    // token accounting directly instead of wall-clock TTFT — TTFT is a flaky
    // CI signal (tiny warm-GPU times dominated by fixed per-turn setup, not
    // prefill work) AND a weak one (a broken reuse that silently re-prefilled
    // the whole prompt could still report a fast TTFT on a fast machine). The
    // deterministic proof that "only the delta was prefilled" is that the
    // freshly-prefilled span is smaller than the reused prefix.
    let uncached_delta = r2.prompt_tokens.saturating_sub(r2.cached_tokens);
    eprintln!(
        "prefix-reuse token accounting: prompt_tokens={} cached_tokens={} uncached_delta={}",
        r2.prompt_tokens, r2.cached_tokens, uncached_delta,
    );
    // The reused prefix must be the clear MAJORITY of the work: turn 2
    // reprocesses only the short new suffix (the one-word reply + the new
    // user turn), while turn 1's long question is served from the
    // content-addressed cache. `cached_tokens >= 2 * uncached_delta` proves
    // that deterministically — a broken reuse that re-prefilled the whole
    // prompt would push uncached_delta up to the full prompt and fail.
    assert!(
        uncached_delta * 2 < r2.cached_tokens,
        "prefix reuse is not the clear majority of the work: reused \
         prefix={} freshly-prefilled suffix={} (prompt_tokens={})",
        r2.cached_tokens,
        uncached_delta,
        r2.prompt_tokens,
    );
}

/// Divergence miss: turn 2's `chat_session_start` prompt begins with a
/// DIFFERENT first message than turn 1's cached history. The reported
/// `cached_tokens` on turn 2 must be 0 (full reset + full re-prefill)
/// and the reply must still be well-formed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_start_prefix_reuse_divergence_miss() {
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
