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

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
use mlx_core::tokenizer::ChatMessage;

/// Clone `src` into a fresh `target/`-rooted dir with the weight files
/// symlinked and `config.json` patched to `use_block_paged_cache=false`,
/// returning the new path. The path is leaked — these run a couple of times
/// per session and `target/` is already a build artifact, so cleanup is
/// best-effort and not needed for correctness.
///
/// The vision-capable MTP checkpoint the MTP-vs-AR oracles run against defaults
/// to the block-paged KV backend at load (its config carries a `vision_config`).
/// The "MTP byte-matches AR" property only holds on the FLAT backend: paged
/// full-attention decode differs from flat by ~1 bf16 ULP, and streaming MTP on
/// a paged model routes through the flat-dense path while AR stays paged — a
/// cross-backend mix that is not byte-comparable. Pin flat so both paths run the
/// same backend.
fn flat_clone_model_dir(src: &Path, suffix: &str) -> Result<PathBuf, String> {
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

    let dst = workspace_target.join(format!("delta-chat-flat-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    // Symlink the (multi-GB) weight files; only config.json is copied + patched.
    for entry in fs::read_dir(src).map_err(|e| format!("read_dir({}): {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        if !from.is_file() {
            continue;
        }
        let to = dst.join(entry.file_name());
        if entry.file_name() == "config.json" {
            fs::copy(&from, &to)
                .map_err(|e| format!("copy({} -> {}): {e}", from.display(), to.display()))?;
        } else {
            std::os::unix::fs::symlink(&from, &to)
                .map_err(|e| format!("symlink({} -> {}): {e}", from.display(), to.display()))?;
        }
    }

    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path)
        .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
    let mut cfg: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(false);
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty)
        .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;

    Ok(dst)
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
    mut rx: tokio::sync::mpsc::UnboundedReceiver<
        napi::Result<mlx_core::engine::types::ChatStreamChunk>,
    >,
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
            None,
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
        is_error: None,
        reasoning_content: None,
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

// ---------------------------------------------------------------------
// Regression: nonpositive `max_new_tokens` budget → 0 generated tokens
// on the MTP decode path, matching the AR `decode_loop!` semantics.
// ---------------------------------------------------------------------
//
// The MTP decode macro (`decode_loop_mtp!` in `mtp_decode.rs`) used to
// UNCONDITIONALLY push the prefill-seed token before its loop's length
// check, so `maxNewTokens == 0` emitted ONE token where AR's
// `for step in 0..max` emits ZERO. A NEGATIVE budget additionally wrapped
// through `as usize` to an effectively unbounded cap, so only EOS /
// repetition / cancellation could ever stop generation. The fix clamps
// the budget (`($max).max(0) as usize`) once and guards the initial emit
// on it, so MTP now matches AR: 0 new tokens for a nonpositive budget.
//
// This test exercises BOTH the MTP-enabled config (the regression) and
// the AR baseline (the parity target).
//
// CAVEAT (and why we no longer "stay green either way"): when
// `enable_mtp = true` but the loaded checkpoint has NO MTP head, the
// engine's gate (`enable_mtp && has_mtp_weights()`) silently falls back
// to the AR `decode_loop!`. In that case the "MTP" assertions below would
// actually re-test the AR path and pass WITHOUT ever entering
// `decode_loop_mtp!` — a false positive. To prevent that, we first probe
// the load-time `has_mtp_weights()` signal AND run a small positive-budget
// MTP generation, confirming via the performance stat
// (`mtp_mean_accepted_tokens`) that the MTP decode path genuinely ran. If
// MTP is NOT active (no MTP head in the checkpoint), we print a skip
// message and `return` rather than running the MTP assertions as if they
// passed.
//
// COVERAGE HONESTY: with an MTP-capable checkpoint this directly catches
// the 1-vs-0 budget regression in `decode_loop_mtp!`. NOT covered here:
// (1) the deterministic pre-cancel-flag sub-case (the non-streaming
//     `chat_session_start` harness can't pre-set a `CancelHandle` before
//     loop entry — see the note further down; the `max_as_usize == 0`
//     short-circuit is placed as the loop's first statement, so the
//     pre-cancelled path is covered by reasoning + statement placement,
//     not a runtime pre-set); and
// (2) the MoE call-site (`decode_loop_mtp!` is also expanded for the MoE
//     model, exercised only by a separate MoE checkpoint).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn nonpositive_budget_emits_zero_tokens_mtp_matches_ar() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/qwen3.5-0.8b-mlx-bf16 with an MTP head)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {}",
        model_path
    );

    let flat_dir = flat_clone_model_dir(model_dir, "nonpos").expect("flat clone failed");
    let model = Qwen3_5Model::load(flat_dir.to_string_lossy().into_owned())
        .await
        .expect("failed to load Qwen3.5 model");

    // Build a config with an explicit MTP toggle and a given budget.
    let cfg_with = |max_new_tokens: i32, enable_mtp: bool| ChatConfig {
        enable_mtp: Some(enable_mtp),
        ..chat_config_default(max_new_tokens)
    };

    // 1) AR baseline at budget 0: empty range → 0 tokens (the parity
    //    target the MTP path must match).
    let ar_zero = model
        .chat_session_start(
            vec![user_message("Say hi in one short word.")],
            Some(cfg_with(0, false)),
        )
        .await
        .expect("AR max_new_tokens=0 chat_session_start failed");
    println!(
        "AR budget=0: num_tokens={} finish_reason={:?}",
        ar_zero.num_tokens, ar_zero.finish_reason
    );
    assert_eq!(
        ar_zero.num_tokens, 0,
        "AR baseline must emit 0 tokens at max_new_tokens=0, got {}",
        ar_zero.num_tokens
    );
    // Pin the AR parity target: AR's empty `0..0` range never observes
    // cancel/EOS/repetition, so finish_reason stays at its "length" init.
    // The MTP assertions below compare against THIS captured baseline
    // (not a hardcoded literal) so the "MTP matches AR" claim is airtight.
    assert_eq!(
        ar_zero.finish_reason, "length",
        "AR baseline must report finish_reason=\"length\" at max_new_tokens=0, got {:?}",
        ar_zero.finish_reason
    );

    // MTP-active gate: confirm the loaded checkpoint actually has an MTP
    // head AND that the MTP decode path genuinely runs before asserting
    // anything about it. Without this, a non-MTP checkpoint would make the
    // budget-0 / negative assertions below silently re-test the AR path
    // (the engine falls back to `decode_loop!` when `has_mtp_weights()` is
    // false) and pass as a FALSE POSITIVE.
    //
    // `has_mtp_weights()` is the load-time snapshot the engine itself uses
    // in its gate (`enable_mtp && has_mtp_weights()`). At a 0 budget no
    // tokens are generated so MTP acceptance is NOT observable; therefore we
    // run a SMALL POSITIVE-budget MTP generation and require the runtime
    // performance stat `mtp_mean_accepted_tokens` to be present — proof the
    // `decode_loop_mtp!` path executed at least one cycle.
    if !model.has_mtp_weights() {
        eprintln!(
            "skipping MTP assertions: checkpoint at {} has no MTP head \
             (has_mtp_weights() == false); the budget-0 assertions would \
             otherwise re-test the AR fallback as a false positive",
            model_path
        );
        return;
    }
    let mtp_probe = model
        .chat_session_start(
            vec![user_message("Say hi in one short word.")],
            Some(cfg_with(8, true)),
        )
        .await
        .expect("MTP positive-budget probe chat_session_start failed");
    let mtp_ran = mtp_probe
        .performance
        .as_ref()
        .and_then(|p| p.mtp_mean_accepted_tokens)
        .is_some();
    println!(
        "MTP probe (budget=8): num_tokens={} mtp_mean_accepted_tokens={:?}",
        mtp_probe.num_tokens,
        mtp_probe
            .performance
            .as_ref()
            .and_then(|p| p.mtp_mean_accepted_tokens)
    );
    if !mtp_ran {
        eprintln!(
            "skipping MTP assertions: has_mtp_weights() is true but a \
             positive-budget MTP run reported no mtp_mean_accepted_tokens \
             (MTP decode path did not execute — e.g. compiled-path / paged \
             gate off); not running the MTP assertions as if they passed"
        );
        return;
    }

    // 2) MTP path at budget 0: this is the regression. Before the fix the
    //    unconditional prefill-seed push made this 1. It must now be 0.
    let mtp_zero = model
        .chat_session_start(
            vec![user_message("Say hi in one short word.")],
            Some(cfg_with(0, true)),
        )
        .await
        .expect("MTP max_new_tokens=0 chat_session_start failed");
    println!(
        "MTP budget=0: num_tokens={} finish_reason={:?}",
        mtp_zero.num_tokens, mtp_zero.finish_reason
    );
    assert_eq!(
        mtp_zero.num_tokens, 0,
        "MTP path must emit 0 tokens at max_new_tokens=0 (matching AR), got {}",
        mtp_zero.num_tokens
    );
    // #3 fix: at a 0 budget the MTP loop must short-circuit to "length"
    // (AR's empty `0..0` range never observes cancel/EOS/repetition, so its
    // finish_reason stays at the "length" init). Before this fix the loop
    // was still entered and the cancelled/EOS checks could run first; with
    // a pre-set cancel flag that produced "cancelled" where AR reports
    // "length". The non-streaming harness here can't pre-set the cancel
    // flag (see the note below), but the non-cancelled finish_reason must
    // still be "length" — and the new `max_as_usize == 0` short-circuit is
    // the same code path that the pre-cancelled case takes.
    assert_eq!(
        mtp_zero.finish_reason, ar_zero.finish_reason,
        "MTP finish_reason at max_new_tokens=0 must match the AR baseline \
         ({:?}), got {:?}",
        ar_zero.finish_reason, mtp_zero.finish_reason
    );
    // Pre-cancelled streaming sub-case (#3): the regression was that a
    // request whose cancel flag is ALREADY set at loop entry, with a 0
    // budget under MTP, reported "cancelled" while AR reports "length".
    // This non-streaming `chat_session_start` harness has no `CancelHandle`
    // wired before the macro is entered (the streaming path sets the flag
    // via `handle.cancel()` AFTER dispatch, which races the decode and
    // cannot be made to land before loop entry deterministically here), so
    // we cannot pre-set the flag in-test. The `max_as_usize == 0`
    // short-circuit is placed as the VERY FIRST statement inside `loop {}`,
    // BEFORE the cancelled check, so a pre-cancelled 0-budget request takes
    // exactly the branch asserted above and yields "length" too. The
    // assertion on `mtp_zero.finish_reason` therefore covers the same code
    // path; the cancel-flag ordering is verified by reasoning + the
    // statement placement rather than a runtime pre-set.

    // 3) Negative budget on the MTP path: previously wrapped via `as
    //    usize` to a huge cap → effectively unbounded. Must now clamp to
    //    0 like AR's empty range.
    let mtp_neg = model
        .chat_session_start(
            vec![user_message("Say hi in one short word.")],
            Some(cfg_with(-5, true)),
        )
        .await
        .expect("MTP max_new_tokens=-5 chat_session_start failed");
    println!(
        "MTP budget=-5: num_tokens={} finish_reason={:?}",
        mtp_neg.num_tokens, mtp_neg.finish_reason
    );
    assert_eq!(
        mtp_neg.num_tokens, 0,
        "MTP path must emit 0 tokens at a negative budget, got {}",
        mtp_neg.num_tokens
    );
    // A negative budget clamps to 0, so it takes the same short-circuit as
    // the 0 case (#3 fix) and must match the same AR baseline.
    assert_eq!(
        mtp_neg.finish_reason, ar_zero.finish_reason,
        "MTP finish_reason at a negative budget (clamped to 0) must match the \
         AR baseline ({:?}), got {:?}",
        ar_zero.finish_reason, mtp_neg.finish_reason
    );
}

// ---------------------------------------------------------------------
// Regression: cancel MID-MTP-CYCLE must not corrupt the next delta turn
// ---------------------------------------------------------------------
//
// The eager-MTP decode commits a whole speculative cycle into the flat
// trunk caches (`self.caches`) BEFORE the per-token emit loop streams the
// accepted tokens out. A cancel BETWEEN those emits strands the committed-
// but-unemitted tail in the cache: `self.caches` ends advanced past the
// saved `cached_token_history`. The flat `rollback_unemitted` closure was a
// no-op (model.rs), so the FOLLOWING delta turn prefilled on top of the
// over-advanced caches → RoPE skew / orphaned K-V → corrupt reply. The fix
// marks the flat caches desynced on a mid-cycle stop so the next turn
// discards them and re-prefills the full history into fresh caches.
//
// ORACLE: an un-cancelled MTP run. The follow-up after a healed cancel must
// match the follow-up of a run that committed the IDENTICAL turn-1 history but
// never desynced. AR is NOT the oracle — speculative MTP and plain AR pick
// different tokens on T=0 argmax near-ties, so an AR reference diverges from
// MTP for reasons unrelated to the heal (verified empirically). Running the
// SAME (MTP) path for both isolates the desync/heal as the only variable, so
// equal-length turn-1 histories are bit-identical greedy prefixes.
//
// Two things the async cancel makes non-deterministic, both handled here:
//   1. HOW MANY turn-1 tokens the cancel commits — the emit loop pushes to
//      history BEFORE its cancel check, and a "cancelled" stop (unlike
//      "length") does not force the trailing-token drop, so the committed
//      count is host-timing-dependent. Read the ACTUAL committed length via
//      `cached_history_len_for_test` and length-match the reference's budget
//      so both runs commit the identical greedy prefix.
//   2. WHETHER the cancel strands u>0 tokens (→ heal) depends on the
//      checkpoint's per-cycle acceptance, which the public API can't force.
//      So this is a GUARD: it never false-fails (u==0 -> both runs warm-
//      continue the same history and agree trivially) and it catches the
//      desync whenever the cancel lands mid-cycle. A counting prompt (high,
//      contiguous MTP acceptance) maximises that chance.
//
// Gated on an MTP-head checkpoint — the desync only exists on the eager-MTP
// path; on a non-MTP checkpoint or if MTP did not actually run it skips.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn cancel_midcycle_then_continue_mtp_matches_uncancelled() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (needs an MTP-head Qwen3.5 Dense checkpoint)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {}",
        model_path
    );

    let flat_dir = flat_clone_model_dir(model_dir, "cancel").expect("flat clone failed");
    let model = Qwen3_5Model::load(flat_dir.to_string_lossy().into_owned())
        .await
        .expect("failed to load Qwen3.5 model");

    // MTP-active gate: the desync only exists on the eager-MTP path. If the
    // checkpoint has no MTP head (or MTP does not actually run) both runs below
    // fall back to AR, neither desyncs, and the comparison is vacuous — skip
    // instead of passing as a false positive. Mirrors the probe in
    // `nonpositive_budget_emits_zero_tokens_mtp_matches_ar`.
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint has no MTP head (has_mtp_weights() == false)");
        return;
    }
    let probe = model
        .chat_session_start(
            vec![user_message("Count from 1 to 12, space separated.")],
            Some(ChatConfig {
                enable_mtp: Some(true),
                ..chat_config_default(16)
            }),
        )
        .await
        .expect("MTP probe chat_session_start failed");
    let mtp_ran = probe
        .performance
        .as_ref()
        .and_then(|p| p.mtp_mean_accepted_tokens)
        .is_some();
    if !mtp_ran {
        eprintln!(
            "skipping: MTP head present but decode_loop_mtp! did not run \
             (mtp_mean_accepted_tokens absent)"
        );
        return;
    }

    // Turn-1 stop policy. The MTP path cancels mid-cycle (the desync trigger);
    // the AR ground truth instead stops cleanly at a fixed new-token budget so
    // it never races the cancel and commits a deterministic history.
    #[derive(Clone, Copy)]
    enum Turn1Stop {
        CancelAfter(usize),
        Budget(usize),
    }

    // Runs turn 1 under `stop`, then a fixed turn-2 follow-up on the resulting
    // caches. Returns (turn-1 streamed-token count, full turn-2 reply text,
    // committed history length after turn 1, whether turn 1 armed the desync
    // heal). The committed length — read from `cached_token_history` between the
    // turns — is how many tokens (prompt + committed generation) the session
    // actually committed, the quantity a mid-cycle cancel makes racy. The desync
    // flag reports whether the cancel actually stranded tokens (so the follow-up
    // heals) or landed clean (so it warm-continues) — coverage, not a false green.
    async fn scenario(
        model: &Qwen3_5Model,
        enable_mtp: bool,
        stop: Turn1Stop,
    ) -> (usize, String, usize, bool) {
        let max_new = match stop {
            Turn1Stop::Budget(n) => n as i32,
            Turn1Stop::CancelAfter(_) => 64,
        };
        let turn1_cfg = ChatConfig {
            enable_mtp: Some(enable_mtp),
            include_reasoning: Some(true),
            ..chat_config_default(max_new)
        };
        let (handle, mut rx) = model
            .chat_stream_session_start_for_test(
                vec![user_message(
                    "Count slowly upward, one number per step: 1 2 3 4 5 and keep going.",
                )],
                Some(turn1_cfg),
            )
            .expect("turn 1 stream dispatch failed");
        // The streaming emit loop fires the callback once per accepted token on
        // BOTH paths, so `emitted` is the saved-history token count (pre
        // drop-last) for either a cancel or a length stop.
        let mut emitted = 0usize;
        while let Some(result) = rx.recv().await {
            let chunk = result.expect("turn 1 stream error");
            if chunk.done {
                break;
            }
            emitted += 1;
            if let Turn1Stop::CancelAfter(k) = stop
                && emitted == k
            {
                handle.cancel();
            }
        }

        // Flat-MTP state, read while the session is idle between turns (the model
        // thread serializes commands, so this observes turn 1 fully finalized):
        // the committed length the reference must reproduce (a mid-cycle cancel
        // commits a host-timing-dependent count), and whether the cancel armed
        // the desync heal.
        let (committed_after_turn1, desynced_after_turn1, _) =
            model.mtp_flat_state_for_test().await;

        // Turn 2: follow-up delta on top of the (possibly desynced) caches.
        let turn2_cfg = ChatConfig {
            enable_mtp: Some(enable_mtp),
            include_reasoning: Some(true),
            ..chat_config_default(24)
        };
        let (_h2, rx2) = model
            .chat_stream_session_continue_for_test(
                "Repeat back, in order, every number you listed so far.".to_string(),
                None,
                Some(turn2_cfg),
            )
            .expect("turn 2 continue dispatch failed");
        let (chunks2, _ttft, done2) = drain_stream_turn(rx2).await;
        assert!(done2, "turn 2 (enable_mtp={enable_mtp}) didn't reach done");
        let full2: String = chunks2.iter().map(|c| c.text.as_str()).collect();
        (emitted, full2, committed_after_turn1, desynced_after_turn1)
    }

    // MTP path: cancel mid-cycle to strand drafted-but-unemitted tokens, the
    // condition the desync heal must repair. Capture its emitted count, the
    // committed turn-1 history length the cancel left behind, and whether it
    // actually armed the heal (`cancel_desynced`).
    let (n_mtp, mtp_turn2, h1_mtp, cancel_desynced) =
        scenario(&model, true, Turn1Stop::CancelAfter(3)).await;
    assert!(
        n_mtp >= 3,
        "MTP turn-1 emitted fewer tokens ({n_mtp}) than the cancel point; cannot \
         exercise a mid-cycle cancel"
    );

    // Reference: an un-cancelled MTP run that commits the SAME turn-1 history.
    // The heal rebuilds the follow-up's context from the session's COMMITTED
    // history, so the fair oracle is a run over the identical committed history
    // that never desynced. The earlier version compared against AR and assumed a
    // mid-cycle cancel commits `n_mtp - 1` tokens the way a length stop does —
    // both wrong, and jointly this test's flakiness: only `finish_reason ==
    // "length"` forces the trailing-token drop (`engine/cache.rs`) while the
    // streaming loop pushes each token into history BEFORE its cancel check
    // (`n_mtp` is counted from the callback AFTER it), so a cancel commits a
    // host-timing-dependent `n_mtp-1 .. n_mtp+1` tokens; and MTP vs AR pick
    // different T=0 near-tie tokens, so an AR reference diverges regardless.
    //
    // Read the committed length instead of guessing. `scenario` returns the
    // actual `cached_token_history` length after turn 1 (`h1_*`), and `Budget(b)`
    // commits exactly `b - 1` generation tokens (length stop drops the last) on
    // top of the fixed turn-1 prompt, so committed length is linear in the
    // budget. Run one reference turn at `Budget(n_mtp)` and, if it committed a
    // different length than the cancel, correct the budget by the measured gap
    // and re-run. Both runs are MTP, so equal committed length ⇒ identical
    // greedy prefix (T=0 speculative decode is exact within its own path).
    let (_n_ref0, ref0_turn2, h1_ref0, _) = scenario(&model, true, Turn1Stop::Budget(n_mtp)).await;
    let (ref_turn2, h1_ref) = if h1_ref0 == h1_mtp {
        // Cancel happened to commit the same length a `Budget(n_mtp)` stop does;
        // the reference run already matches — no correction turn needed.
        (ref0_turn2, h1_ref0)
    } else {
        // b* = n_mtp + (h1_mtp − h1_ref0): shifting the budget by the committed-
        // length gap shifts the committed history by the same amount, so the
        // reference commits exactly the history the cancel left.
        let budget_star = n_mtp as i64 + h1_mtp as i64 - h1_ref0 as i64;
        assert!(
            budget_star >= 2,
            "computed reference budget {budget_star} too small to commit a turn-1 \
             history (n_mtp={n_mtp}, h1_mtp={h1_mtp}, h1_ref0={h1_ref0})"
        );
        let (_n_ref, t2, h1, _) =
            scenario(&model, true, Turn1Stop::Budget(budget_star as usize)).await;
        (t2, h1)
    };

    // Coverage, not a false green: whether THIS cancel landing armed the heal
    // (`cancel_desynced`) is host-timing-dependent. When true, the equality below
    // exercises the discard+re-prefill path; when false the cancel landed clean
    // and both runs warm-continue. `desync_heal_reprefills_to_uncancelled` covers
    // the heal deterministically, so this test never needs to force the landing.
    println!(
        "cancel: emitted={n_mtp} committed_hist={h1_mtp} desynced={cancel_desynced}  \
         ref cal={h1_ref0}  ref matched={h1_ref}"
    );
    println!("cancel turn2 = {mtp_turn2:?}");
    println!("ref    turn2 = {ref_turn2:?}");

    // Precondition for a fair comparison: the budget correction reproduced the
    // exact committed turn-1 history the cancel left. If this fails the readout
    // or the model misbehaved — a distinct failure from a broken heal.
    assert_eq!(
        h1_ref, h1_mtp,
        "budget-matched reference run did not reproduce the cancel's committed \
         history length (h1_ref={h1_ref}, h1_mtp={h1_mtp}); cannot compare turn-2 fairly"
    );

    // KEY: over the identical committed history, a healed cancel yields exactly
    // the un-cancelled reply. Under the original bug the flat caches were advanced
    // past the committed history (stranded drafted tokens never rolled back), so
    // this follow-up diverges grossly. Asserted unconditionally — no skip.
    assert_eq!(
        mtp_turn2, ref_turn2,
        "MTP follow-up after a mid-cycle cancel diverged from the un-cancelled \
         reply over the SAME committed history — flat-cache desync not healed.\n\
         cancel={mtp_turn2:?}\nref={ref_turn2:?}"
    );
}

// ---------------------------------------------------------------------
// Regression: the desync HEAL must re-prefill to the un-cancelled reply
// ---------------------------------------------------------------------
//
// Companion to `cancel_midcycle_then_continue_mtp_matches_uncancelled`. That
// test only exercises the heal when a mid-cycle cancel happens to strand
// tokens, which is host-timing-dependent (a fast host lands every cancel on a
// clean boundary, so the follow-up warm-continues and the comparison never
// touches the heal). This test arms the heal DETERMINISTICALLY via
// `force_flat_mtp_desync_for_test` and proves the discard+re-prefill path
// (model.rs, `if self.flat_mtp_caches_desynced { .. }`) reproduces the
// un-cancelled reply. The heal re-prefills from `cached_token_history` and
// discards the caches, so arming the flag on a clean session is faithful: the
// heal's OUTPUT depends only on the committed history, not on what the mid-cycle
// stop left in the (discarded) caches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn desync_heal_reprefills_to_uncancelled() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (needs an MTP-head Qwen3.5 Dense checkpoint)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {model_path}"
    );
    let flat_dir = flat_clone_model_dir(model_dir, "heal").expect("flat clone failed");
    let model = Qwen3_5Model::load(flat_dir.to_string_lossy().into_owned())
        .await
        .expect("failed to load Qwen3.5 model");
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint has no MTP head (has_mtp_weights() == false)");
        return;
    }

    // One turn-1 at a fixed budget (identical, clean history on both runs), then
    // a fixed turn-2. `arm_desync` forces the follow-up down the heal path.
    // Returns (committed turn-1 length, turn-2 reply, whether the heal actually
    // ran). Heal detection uses the `full_reprefill_count` from the state
    // snapshot — it increments only when turn 2 takes the discard+re-prefill
    // path — because the streaming chunk's `prompt_tokens`/`cached_tokens` report
    // identically for the heal and warm arms and cannot distinguish them.
    async fn run(model: &Qwen3_5Model, budget: i32, arm_desync: bool) -> (usize, String, bool) {
        let cfg1 = ChatConfig {
            enable_mtp: Some(true),
            include_reasoning: Some(true),
            ..chat_config_default(budget)
        };
        let (_h, mut rx) = model
            .chat_stream_session_start_for_test(
                vec![user_message(
                    "Count slowly upward, one number per step: 1 2 3 4 5 and keep going.",
                )],
                Some(cfg1),
            )
            .expect("turn 1 dispatch failed");
        while let Some(r) = rx.recv().await {
            if r.expect("turn 1 stream error").done {
                break;
            }
        }
        let (committed, desynced0, reprefills_before) = model.mtp_flat_state_for_test().await;
        assert!(
            !desynced0,
            "a clean length-stopped turn 1 must not be desynced"
        );
        if arm_desync {
            model.force_flat_mtp_desync_for_test().await;
            let (_, armed, _) = model.mtp_flat_state_for_test().await;
            assert!(armed, "force_flat_mtp_desync_for_test did not arm the flag");
        }

        let cfg2 = ChatConfig {
            enable_mtp: Some(true),
            include_reasoning: Some(true),
            ..chat_config_default(24)
        };
        let (_h2, rx2) = model
            .chat_stream_session_continue_for_test(
                "Repeat back, in order, every number you listed so far.".to_string(),
                None,
                Some(cfg2),
            )
            .expect("turn 2 dispatch failed");
        let (chunks2, _ttft, done2) = drain_stream_turn(rx2).await;
        assert!(done2, "turn 2 didn't reach done");
        let text: String = chunks2.iter().map(|c| c.text.as_str()).collect();
        let (_, desynced_after, reprefills_after) = model.mtp_flat_state_for_test().await;
        // Heal ran iff turn 2 took the discard+re-prefill path (the counter
        // incremented). The streaming chunk's `prompt_tokens`/`cached_tokens`
        // report identically for heal and warm, so the counter is the only
        // observable signal. The flag is always cleared post-turn, so
        // `desynced_after` is expected false whether or not the heal fired.
        assert!(
            !desynced_after,
            "the desync flag must be cleared after a turn"
        );
        let healed = reprefills_after > reprefills_before;
        (committed, text, healed)
    }

    let budget = 8;
    let (h_heal, t_heal, healed) = run(&model, budget, true).await;
    let (h_ref, t_ref, _) = run(&model, budget, false).await;

    println!("heal: committed={h_heal} healed={healed}");
    println!("heal turn2 = {t_heal:?}");
    println!("ref  turn2 = {t_ref:?}");

    // The two runs share turn 1 (same budget, both clean MTP) so they commit the
    // identical greedy prefix — a fair basis for comparing turn 2.
    assert_eq!(
        h_heal, h_ref,
        "same budget must commit the same turn-1 history length"
    );
    // Non-vacuous: the armed run genuinely took the discard+re-prefill heal path.
    assert!(
        healed,
        "forced desync did not take the re-prefill heal path (committed={h_heal})"
    );
    // KEY: the heal re-prefills to exactly the un-cancelled warm reply.
    assert_eq!(
        t_heal, t_ref,
        "desync-heal re-prefill diverged from the un-cancelled warm reply over the \
         SAME committed history.\nheal={t_heal:?}\nref ={t_ref:?}"
    );
}
