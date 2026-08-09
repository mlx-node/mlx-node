//! Real-weights mid-prefill cancellation gate for Qwen3 (hazard H1b,
//! fail closed).
//!
//! A cancel flag flipped MID-PREFILL must abort the prefill at the next
//! chunk boundary (`MLX_PAGED_PREFILL_CHUNK_SIZE`-sized chunks) instead
//! of running the whole prompt to completion, and the aborted turn must
//! fail CLOSED: the paged request is released un-registered, no session
//! history is saved, so the next identical turn re-prefills from zero
//! (`cached_tokens == 0`).
//!
//! Named mutation this gate catches:
//!   * checkpoints reverted (HEAD behavior): the first turn completes the
//!     ENTIRE prefill before the decode-loop cancel fires — the timing
//!     assertion fails (and the completed turn registers the prompt
//!     blocks, so `cached_tokens > 0` fails too);
//!   * checkpoints present but fail-OPEN (abort path skipped / history
//!     still saved): the second turn warm-hits partial state —
//!     `cached_tokens == 0` fails.
//!
//! Pattern: `qwen3_paged_vs_flat_parity.rs` — gated on
//! `MLX_TEST_MODEL_PATH` so a plain `cargo test` (and `--ignored` without
//! the env var) still passes; a missing fixture SKIPS, never fails.
//!
//! Run locally with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/qwen3-0.6b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_prefill_cancel \
//!     -- --ignored --nocapture
//! ```

use std::path::Path;
use std::time::{Duration, Instant};

use mlx_core::engine::types::{ChatConfig, ChatStreamChunk};
use mlx_core::models::qwen3::persistence::load_with_thread as qwen3_load_with_thread;
use mlx_core::tokenizer::ChatMessage;

/// Chunked-prefill chunk size for this gate. 512 tokens per chunk over a
/// multi-thousand-token prompt gives the cancel timer many chunk
/// boundaries to land on.
const PREFILL_CHUNK_SIZE: &str = "512";

/// Delay before the timer thread flips the cancel flag. Must be long
/// enough that the model thread has dequeued the turn and entered the
/// prefill (the pre-start cancel guard would otherwise swallow the turn
/// before any chunk ran — asserted below), and short enough to land well
/// before the multi-second full prefill completes.
const CANCEL_DELAY_MS: u64 = 150;

fn user_message(content: String) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: None,
        audio: None,
    }
}

/// A prompt long enough for many 512-token prefill chunks (~5.5k tokens
/// on the qwen3 tokenizer — 10+ chunk boundaries).
fn long_prompt() -> String {
    let paragraph = "The quick brown fox jumps over the lazy dog while the \
                     patient owl watches from a very tall oak tree near the \
                     river bend, counting every leaf that falls into the \
                     slow-moving water below. ";
    let mut prompt = String::with_capacity(paragraph.len() * 150 + 64);
    prompt.push_str("Summarize the following text in one sentence:\n");
    for _ in 0..150 {
        prompt.push_str(paragraph);
    }
    prompt
}

fn chat_config(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        reuse_cache: Some(true),
        report_performance: Some(true),
        include_reasoning: Some(true),
        ..Default::default()
    }
}

/// Outcome of draining one streaming turn to its terminal signal.
struct TurnOutcome {
    /// First error item delivered through the sink, if any.
    error: Option<String>,
    /// Terminal done-chunk, if the turn completed normally.
    done: Option<ChatStreamChunk>,
    /// Elapsed from dispatch to the terminal signal (error or done).
    elapsed: Duration,
    /// Elapsed from dispatch to the FIRST item of any kind — for a
    /// normally-completing turn this is the TTFT proxy (chunks only
    /// start once the whole prefill finished).
    first_item: Option<Duration>,
}

async fn drain_turn(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<napi::Result<ChatStreamChunk>>,
    started: Instant,
) -> TurnOutcome {
    let mut outcome = TurnOutcome {
        error: None,
        done: None,
        elapsed: Duration::ZERO,
        first_item: None,
    };
    let mut terminal_seen = false;
    while let Some(item) = rx.recv().await {
        if outcome.first_item.is_none() {
            outcome.first_item = Some(started.elapsed());
        }
        match item {
            Err(e) => {
                if !terminal_seen {
                    outcome.error = Some(e.reason.clone());
                    outcome.elapsed = started.elapsed();
                    terminal_seen = true;
                }
            }
            Ok(chunk) => {
                if chunk.done && !terminal_seen {
                    outcome.elapsed = started.elapsed();
                    terminal_seen = true;
                    outcome.done = Some(chunk);
                }
            }
        }
    }
    if !terminal_seen {
        // Channel closed without a terminal signal (model thread
        // dropped the sink) — still record when.
        outcome.elapsed = started.elapsed();
    }
    outcome
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3 checkpoint"]
async fn qwen3_paged_mid_prefill_cancel_fails_closed() {
    // Chunk the paged prefill BEFORE anything can initialize the
    // process-wide OnceLock that caches this env var. This test file is
    // its own test binary and this is its only test, so nothing races
    // the read.
    //
    // SAFETY: no other thread exists yet (the model thread spawns at
    // load, below) and nothing has read the variable.
    unsafe { std::env::set_var("MLX_PAGED_PREFILL_CHUNK_SIZE", PREFILL_CHUNK_SIZE) };

    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MODEL_PATH does not exist: {model_path}",
    );

    let model = qwen3_load_with_thread(&model_path)
        .await
        .expect("failed to load Qwen3 model");
    assert!(
        model.has_block_paged_cache(),
        "this gate exercises the PAGED chunked prefill; the checkpoint \
         must load with use_block_paged_cache enabled (qwen3's default)",
    );

    // `ChatMessage` is not `Clone` (napi payload buffers) — rebuild the
    // identical message list per turn instead.
    let messages = || vec![user_message(long_prompt())];

    // ---- Turn 1: cancel mid-prefill via a timer thread. ----
    let started1 = Instant::now();
    let (handle1, rx1) = model
        .chat_stream_session_start_for_test(messages(), Some(chat_config(64)))
        .expect("turn 1 dispatch failed");
    let timer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(CANCEL_DELAY_MS));
        handle1.cancel();
    });
    let turn1 = drain_turn(rx1, started1).await;
    timer.join().expect("cancel timer thread panicked");

    // Premise guard: the flag must have flipped AFTER the model thread
    // dequeued the turn (i.e. mid-turn), not before — a pre-start cancel
    // proves nothing about chunk-boundary checkpoints.
    if let Some(err) = &turn1.error {
        assert!(
            !err.contains("cancelled before start"),
            "cancel timer fired before the model thread dequeued the \
             turn; raise CANCEL_DELAY_MS (turn 1 error: {err})",
        );
    }
    println!(
        "turn 1: elapsed={:?} error={:?} done_finish_reason={:?}",
        turn1.elapsed,
        turn1.error,
        turn1.done.as_ref().and_then(|c| c.finish_reason.clone()),
    );

    // ---- Turn 2: identical prompt, no cancel — must run to done. ----
    let started2 = Instant::now();
    let (_handle2, rx2) = model
        .chat_stream_session_start_for_test(messages(), Some(chat_config(8)))
        .expect("turn 2 dispatch failed");
    let turn2 = drain_turn(rx2, started2).await;
    assert!(
        turn2.error.is_none(),
        "turn 2 must complete normally (a stale turn-1 cancel flag on the \
         backend would spuriously cancel it): {:?}",
        turn2.error,
    );
    let done2 = turn2.done.expect("turn 2 must reach a terminal done chunk");
    let ttft2 = turn2
        .first_item
        .expect("turn 2 delivered chunks, so first_item must be set");
    println!(
        "turn 2: elapsed={:?} ttft={:?} prompt_tokens={:?} cached_tokens={:?}",
        turn2.elapsed, ttft2, done2.prompt_tokens, done2.cached_tokens,
    );

    // Premise: the prompt really spans many prefill chunks.
    let prompt_tokens = done2
        .prompt_tokens
        .expect("terminal chunk carries prompt_tokens");
    assert!(
        prompt_tokens >= 2048,
        "premise broken: prompt only {prompt_tokens} tokens — not a \
         multi-chunk prefill at chunk size {PREFILL_CHUNK_SIZE}",
    );

    // ---- Gate 1 (H1b responsiveness): the cancel took effect within
    // ~one chunk of prefill work, not after the whole prefill. Turn 2's
    // TTFT bounds a full-prefill duration from above on this machine
    // (cold when fail-closed holds; even warmer at HEAD — which only
    // makes the bound tighter and the assertion redder).
    assert!(
        turn1.elapsed < ttft2.mul_f64(0.6),
        "cancel did not take effect within one prefill chunk: cancelled \
         turn ran {:?} vs full-prefill TTFT {:?} — mid-prefill cancel \
         checkpoints are missing (H1b)",
        turn1.elapsed,
        ttft2,
    );

    // ---- Gate 2 (fail closed): the cancelled prefill must NOT have
    // left partially-written KV registered as a live/reusable prefix —
    // the identical second turn re-prefills from zero.
    assert_eq!(
        done2.cached_tokens,
        Some(0),
        "cancelled prefill leaked a reusable prefix: turn 2 reported \
         cached_tokens={:?} (expected 0 — the aborted paged request must \
         be released without registering its blocks and without saving \
         session history)",
        done2.cached_tokens,
    );
}
