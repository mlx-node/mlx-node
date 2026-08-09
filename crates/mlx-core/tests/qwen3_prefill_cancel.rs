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

use std::fs;
use std::path::{Path, PathBuf};
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

fn assistant_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
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

/// Clone the checkpoint dir with `use_block_paged_cache` forced OFF so the
/// chat turn takes the FLAT engine path (`ChatBackend::prefill` →
/// `flat_prefill`'s 2048-token chunk loop). Weight files are symlinked;
/// only `config.json` is copied and patched. Pattern:
/// `qwen3_paged_vs_flat_parity.rs::clone_model_dir`.
fn clone_flat_model_dir(src: &Path) -> Result<PathBuf, String> {
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
    let dst = workspace_target.join(format!("prefill-cancel-flat-{pid}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;
    let read_dir = fs::read_dir(src).map_err(|e| format!("read_dir({}): {e}", src.display()))?;
    for entry in read_dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_file() {
            if entry.file_name() == "config.json" {
                fs::copy(&from, &to)
                    .map_err(|e| format!("copy({} -> {}): {e}", from.display(), to.display()))?;
            } else {
                std::os::unix::fs::symlink(&from, &to)
                    .map_err(|e| format!("symlink({} -> {}): {e}", from.display(), to.display()))?;
            }
        }
    }
    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path).map_err(|e| format!("read config.json: {e}"))?;
    let mut cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse config.json: {e}"))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(false);
    fs::write(
        &cfg_path,
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?,
    )
    .map_err(|e| format!("write config.json: {e}"))?;
    Ok(dst)
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
    // A missing fixture SKIPS (plan-mandated), never fails the gate.
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: MLX_TEST_MODEL_PATH does not exist: {model_path}");
        return;
    }

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

    // ---- Gate 2 (distinguished error): the cancelled turn must have
    // terminated with the mid-prefill cancel error specifically — an
    // unrelated early failure (load/OOM/template) must not satisfy this
    // gate.
    assert!(
        turn1
            .error
            .as_deref()
            .is_some_and(|e| e.contains("prefill cancelled")),
        "turn 1 must terminate with the distinguished \"prefill \
         cancelled\" error (got error={:?}, done={:?})",
        turn1.error,
        turn1.done.as_ref().and_then(|c| c.finish_reason.clone()),
    );

    // ---- Gate 3 (fail closed): the cancelled prefill must NOT have
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

/// FLAT-path twin of the paged gate: a cancel flipped during an EARLY
/// flat prefill chunk must abort BEFORE the final remainder forward and
/// fail closed (no `save_cache_state`, session invalidated).
///
/// Named mutation this pins (review Finding 1): the flat chunk loops
/// poll only at the top of NON-final iterations — a flag flipped during
/// the last looped chunk was never seen again, the remainder ran, decode
/// treated the cancel as a normal `finish_reason="cancelled"` finish,
/// and `save_cache_state` committed the turn. With the fix, a poll runs
/// before the final remainder whenever an earlier chunk was processed
/// (offset > 0); offset-zero single-shot prefills stay uncancellable.
///
/// The ~5.5k-token prompt spans two full 2048-token loop chunks plus a
/// remainder; the timer fires mid-chunk-1 (chunk ≈ tens of seconds in a
/// debug build), so at the defect the only later poll would be decode's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3 checkpoint"]
async fn qwen3_flat_mid_prefill_cancel_fails_closed() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };
    // A missing fixture SKIPS (plan-mandated), never fails the gate.
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: MLX_TEST_MODEL_PATH does not exist: {model_path}");
        return;
    }

    let flat_dir = clone_flat_model_dir(Path::new(&model_path))
        .expect("failed to clone a flat (use_block_paged_cache=false) model dir");
    let model = qwen3_load_with_thread(&flat_dir.to_string_lossy())
        .await
        .expect("failed to load flat Qwen3 model");
    assert!(
        !model.has_block_paged_cache(),
        "flat clone must load WITHOUT the paged adapter",
    );

    let messages = || vec![user_message(long_prompt())];

    // Turn 1: cancel mid-prefill via a timer thread. The flat fused
    // forward is FAST (prefill ends ~380ms for this whole 5.5k-token
    // prompt on an M5 in a debug build; boundary polls sit at ~0/~130/
    // ~240ms), so the default 100ms lands inside an early chunk and is
    // caught at the NEXT boundary poll — well clear of the uncancellable
    // final-remainder forward, on this machine and on anything slower.
    // `MLX_TEST_FLAT_CANCEL_DELAY_MS` overrides for machine-speed tuning
    // (200ms was the mutation-A/B point aimed between the last loop-top
    // poll and the pre-remainder poll).
    let delay_ms: u64 = std::env::var("MLX_TEST_FLAT_CANCEL_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let started1 = Instant::now();
    let (handle1, rx1) = model
        .chat_stream_session_start_for_test(messages(), Some(chat_config(16)))
        .expect("flat turn 1 dispatch failed");
    let timer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(delay_ms));
        handle1.cancel();
    });
    let turn1 = drain_turn(rx1, started1).await;
    timer.join().expect("cancel timer thread panicked");

    // Premise guard: the flag flipped AFTER the model thread dequeued
    // the turn.
    if let Some(err) = &turn1.error {
        assert!(
            !err.contains("cancelled before start"),
            "cancel timer fired before the model thread dequeued the \
             turn; raise the delay (turn 1 error: {err})",
        );
    }
    println!(
        "flat turn 1: elapsed={:?} error={:?} done_finish_reason={:?}",
        turn1.elapsed,
        turn1.error,
        turn1.done.as_ref().and_then(|c| c.finish_reason.clone()),
    );

    // Gate A (distinguished error, chunk-boundary abort): the turn must
    // terminate with "prefill cancelled" and NO done chunk. At the
    // defect the whole prefill runs, decode cancels normally, and a done
    // chunk (finish_reason="cancelled") arrives instead.
    assert!(
        turn1
            .error
            .as_deref()
            .is_some_and(|e| e.contains("prefill cancelled")),
        "flat turn 1 must abort with the distinguished \"prefill \
         cancelled\" error (got error={:?}, done_finish_reason={:?})",
        turn1.error,
        turn1.done.as_ref().and_then(|c| c.finish_reason.clone()),
    );
    assert!(
        turn1.done.is_none(),
        "flat turn 1 must not emit a terminal done chunk after a \
         mid-prefill cancel",
    );

    // Gate B (fail closed, no cache save): the aborted turn must NOT
    // have registered a live session. A role-aware continue therefore
    // hits the "requires an initialized session" guard. At the defect
    // the cancelled turn SAVED state, a live session exists, and the
    // continue is admitted.
    let mut history = messages();
    history.push(assistant_message("partial reply"));
    history.push(user_message("Continue.".to_string()));
    let (_handle2, rx2) = model
        .chat_stream_session_continue_for_test(history, Some(chat_config(4)))
        .expect("continue dispatch failed");
    let turn2 = drain_turn(rx2, Instant::now()).await;
    assert!(
        turn2
            .error
            .as_deref()
            .is_some_and(|e| e.contains("requires an initialized session")),
        "the cancelled flat prefill must leave NO live session (expected \
         the initialized-session guard, got error={:?} done={:?})",
        turn2.error,
        turn2.done.as_ref().and_then(|c| c.finish_reason.clone()),
    );
}
