//! Gated end-to-end tests for Gemma4 DSpark speculative decoding.
//!
//! Loads ONE Gemma4 target model per test WITH the DSpark draft attached
//! (`Gemma4LoadOptions::draft_model_path`); the plain-AR oracle runs on the
//! SAME instance with `enableMtp: false` (the draft never touches the target
//! weights or the flat AR path, so this is the exact byte-parity reference —
//! and only one ~24 GB target is resident at a time).
//!
//! PRIMARY ORACLE: speculative decoding must be LOSSLESS at T=0 — the
//! DSpark turn's text/raw_text/finish_reason must byte-match the AR run.
//!
//! # Fixture selection: bf16 near-tie robustness (READ BEFORE EDITING PROMPTS)
//!
//! The verify forward evaluates T=1+L rows with the SAME chunked-prefill
//! kernels the AR prompt prefill uses, but the AR DECODE evaluates each
//! token with the single-token (T=1) kernels. Per-row bf16 results differ
//! between the two kernel shapes by ~1 ULP (the effect gemma4's own
//! `prefill_body_gemma4` doc documents — it is why AR prefill splits the
//! last prompt token out). At a position whose top-2 softcapped logits are
//! within ~1-2 bf16 ULP (0.125-0.25 at |logit|≈20), the greedy argmax can
//! legitimately differ between the shapes: measured directly (stock
//! forwards only, zero DSpark code), a 200-step AR trajectory on an
//! open-prose prompt hit exactly one such flip — single-token gap 0.25 (2
//! ULP), batched row top-2 EXACTLY tied — and depth-1/depth-2/depth-7
//! DSpark runs all byte-agree with each other while flipping that one
//! near-tie against AR. The repo already accepts this class as inherent
//! (`paged_decode_long_context_1ulp`: vLLM never asserts bitwise; the
//! qwen3.5 MTP heal de-flake: "AR diverges at T=0 near-ties").
//!
//! The byte-equal oracle is therefore kept STRICT and the fixtures are
//! chosen to be tie-free: constrained generations (counting, single-word
//! answers, recipes) whose greedy top-2 gaps are far above kernel noise.
//! Every fixture below was screened to be byte-equal AR-vs-DSpark on this
//! checkpoint; MLX runs are deterministic, so green is stable per
//! machine + MLX pin. If one of these tests diverges after an MLX bump,
//! check whether the divergence point is a near-tie (top-2 gap <= ~2 bf16
//! ULP → re-screen the fixture) before suspecting the DSpark wiring; a
//! REAL bookkeeping bug (positions, rollback offsets, masks) diverges
//! grossly and immediately, not at a single near-tied token.
//!
//! Env (both required; unset → skip-with-message):
//!   MLX_TEST_GEMMA4_MODEL_PATH   — bf16 unified 48L Gemma-4-12B-IT checkout
//!   MLX_TEST_GEMMA4_DSPARK_PATH  — dspark_gemma4_12b_block7 draft checkout
//!
//! Run (single-threaded is MANDATORY — concurrent model tests oversubscribe
//! the GPU and SIGABRT):
//!
//! ```shell
//! PATH=/usr/bin:$PATH SDKROOT=$(xcrun --show-sdk-path) \
//! MLX_TEST_GEMMA4_MODEL_PATH=/abs/path/to/gemma-4-12b-it \
//! MLX_TEST_GEMMA4_DSPARK_PATH=/abs/path/to/dspark_gemma4_12b_block7 \
//!     cargo test -p mlx-core --test gemma4_dspark --release -- \
//!     --ignored --nocapture --test-threads=1
//! ```

use std::path::Path;

use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::gemma4::model::{Gemma4LoadOptions, Gemma4Model};
use mlx_core::tokenizer::ChatMessage;

const SKIP_MSG: &str = "skipping: set MLX_TEST_GEMMA4_MODEL_PATH (bf16 Gemma-4-12B-IT dir) and \
     MLX_TEST_GEMMA4_DSPARK_PATH (dspark_gemma4_12b_block7 dir) to run the DSpark e2e suite";

fn env_paths() -> Option<(String, String)> {
    let (Ok(model), Ok(draft)) = (
        std::env::var("MLX_TEST_GEMMA4_MODEL_PATH"),
        std::env::var("MLX_TEST_GEMMA4_DSPARK_PATH"),
    ) else {
        eprintln!("{SKIP_MSG}");
        return None;
    };
    assert!(
        Path::new(&model).exists(),
        "MLX_TEST_GEMMA4_MODEL_PATH does not exist: {model}"
    );
    assert!(
        Path::new(&draft).exists(),
        "MLX_TEST_GEMMA4_DSPARK_PATH does not exist: {draft}"
    );
    Some((model, draft))
}

async fn load_dspark_model(model: &str, draft: &str) -> Gemma4Model {
    let m = Gemma4Model::load(
        model.to_string(),
        Some(Gemma4LoadOptions {
            draft_model_path: Some(draft.to_string()),
        }),
    )
    .await
    .expect("Gemma4Model::load with draft_model_path failed");
    assert!(
        m.has_mtp_weights(),
        "draft loaded via Gemma4LoadOptions must flip hasMtpWeights()"
    );
    assert!(
        !m.has_block_paged_cache(),
        "a DSpark load must force the flat KV path (paged adapter off)"
    );
    m
}

fn chat_config(max_new_tokens: i32, enable_mtp: bool) -> ChatConfig {
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
        enable_mtp: Some(enable_mtp),
        mtp_depth: None,
        mtp_adaptive_depth: Some(false),
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

/// Assert the DSpark run byte-matches the AR reference AND actually ran
/// DSpark cycles (not a silent AR fallback), then print the headline stats.
fn assert_matches_ar(label: &str, dspark: &ChatResult, ar: &ChatResult) {
    assert_eq!(
        dspark.text, ar.text,
        "[{label}] DSpark text diverged from AR at T=0"
    );
    assert_eq!(
        dspark.raw_text, ar.raw_text,
        "[{label}] DSpark raw_text diverged from AR at T=0"
    );
    assert_eq!(
        dspark.finish_reason, ar.finish_reason,
        "[{label}] DSpark finish_reason diverged from AR"
    );
    assert_eq!(
        dspark.num_tokens, ar.num_tokens,
        "[{label}] DSpark token count diverged from AR"
    );
    let perf = dspark
        .performance
        .as_ref()
        .expect("DSpark performance metrics missing (gemma4 always reports)");
    let cycles = perf.mtp_cycles.unwrap_or(0);
    assert!(
        cycles > 0,
        "[{label}] expected DSpark cycles to run, got mtp_cycles={cycles:?} (silent AR fallback?)"
    );
    assert!(
        perf.mtp_mean_accepted_tokens_total.is_some(),
        "[{label}] mtp_mean_accepted_tokens_total must be filled after a DSpark turn"
    );
    let ar_perf = ar.performance.as_ref().expect("AR performance missing");
    println!(
        "[{label}] tokens={} cycles={} mean_accepted_total={:?} mean_depth={:?} | decode tok/s: dspark={:.1} ar={:.1}",
        dspark.num_tokens,
        cycles,
        perf.mtp_mean_accepted_tokens_total,
        perf.mtp_mean_depth,
        perf.decode_tokens_per_second,
        ar_perf.decode_tokens_per_second,
    );
}

// ---------------------------------------------------------------------------
// 1. PRIMARY ORACLE: greedy multi-cycle matches-AR
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (real 12B + draft)"]
async fn dspark_greedy_matches_ar_multi_cycle() {
    let Some((model_path, draft_path)) = env_paths() else {
        return;
    };
    let model = load_dspark_model(&model_path, &draft_path).await;

    // Tie-screened fixture (see the module doc): a constrained numbered
    // recipe holds 200 greedy tokens without a near-tie.
    const PROMPT: &str = "Give a simple recipe for pancakes with numbered steps.";

    let ar = model
        .chat_session_start(vec![user_message(PROMPT)], Some(chat_config(200, false)))
        .await
        .expect("AR chat_session_start failed");
    let dspark = model
        .chat_session_start(vec![user_message(PROMPT)], Some(chat_config(200, true)))
        .await
        .expect("DSpark chat_session_start failed");

    assert_matches_ar("multi_cycle", &dspark, &ar);
    let perf = dspark.performance.as_ref().expect("perf missing");
    assert!(
        perf.mtp_cycles.unwrap_or(0) > 1,
        "expected multiple DSpark cycles over 200 tokens"
    );
}

// ---------------------------------------------------------------------------
// 2. RotatingKVCache trap: sliding-window wrap during decode AND prefill
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (real 12B + draft)"]
async fn dspark_greedy_matches_ar_across_sliding_wrap() {
    let Some((model_path, draft_path)) = env_paths() else {
        return;
    };
    let model = load_dspark_model(&model_path, &draft_path).await;

    let paragraph = "The expedition crossed the ridge at dawn, cataloguing mosses, lichens, \
         glacial striations, and the slow braided rivers below, while the cartographer argued \
         with the botanist about the correct name for a small blue flower none of them had \
         seen before. ";

    // (a) Sub-window prompt (~832 tokens) + 600-token budget: the
    // 1024-token sliding window wraps MID-DECODE, so verify blocks append
    // to (and partially roll back from) rotated RotatingKVCache state.
    // The decode tail is a constrained count (tie-free; module doc).
    let mut mid_prompt = String::new();
    for _ in 0..15 {
        mid_prompt.push_str(paragraph);
    }
    mid_prompt.push_str(
        "\n\nIgnore the text above entirely. Count from 1 to 150, one number per line, \
         digits only, no other words. Start immediately with 1.",
    );
    let ar_a = model
        .chat_session_start(
            vec![user_message(&mid_prompt)],
            Some(chat_config(600, false)),
        )
        .await
        .expect("AR (decode wrap) failed");
    assert!(
        ar_a.prompt_tokens < 1024,
        "decode-wrap fixture must START below the sliding window, got {} prompt tokens",
        ar_a.prompt_tokens
    );
    assert!(
        ar_a.prompt_tokens + ar_a.num_tokens > 1024 + 128,
        "decode-wrap fixture must decode WELL past the 1024-token sliding window \
         (prompt {} + generated {})",
        ar_a.prompt_tokens,
        ar_a.num_tokens
    );
    let dspark_a = model
        .chat_session_start(
            vec![user_message(&mid_prompt)],
            Some(chat_config(600, true)),
        )
        .await
        .expect("DSpark (decode wrap) failed");
    assert_matches_ar("decode_wrap", &dspark_a, &ar_a);

    // (b) >1100-token prompt + 300-token budget: the window wraps DURING
    // PREFILL, so the tapped chunked prefill and every verify block run on
    // already-rotated sliding caches.
    let mut long_prompt = String::new();
    for _ in 0..60 {
        long_prompt.push_str(paragraph);
    }
    long_prompt.push_str(
        "\n\nIgnore the text above entirely. Count from 1 to 100, one number per line, \
         digits only, no other words.",
    );
    let ar_b = model
        .chat_session_start(
            vec![user_message(&long_prompt)],
            Some(chat_config(300, false)),
        )
        .await
        .expect("AR (prefill wrap) failed");
    assert!(
        ar_b.prompt_tokens > 1100,
        "prefill-wrap fixture must exceed the sliding window during prefill, got {} tokens",
        ar_b.prompt_tokens
    );
    let dspark_b = model
        .chat_session_start(
            vec![user_message(&long_prompt)],
            Some(chat_config(300, true)),
        )
        .await
        .expect("DSpark (prefill wrap) failed");
    assert_matches_ar("prefill_wrap", &dspark_b, &ar_b);
}

// ---------------------------------------------------------------------------
// 3. Early EOS stop parity + warm-continue delta turn
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (real 12B + draft)"]
async fn dspark_stop_mid_block_then_delta_turn() {
    let Some((model_path, draft_path)) = env_paths() else {
        return;
    };
    let model = load_dspark_model(&model_path, &draft_path).await;

    // A prompt that stops WELL before the budget, so the EOS lands inside a
    // speculative block (either as an accepted draft or as the boundary).
    const PROMPT: &str = "What is the capital of France? Answer with just the city name.";
    const FOLLOW_UP: &str = "And of Italy? Same format.";

    // AR baseline: turn 1 (fresh) + turn 2 (warm continue on the session).
    let ar1 = model
        .chat_session_start(vec![user_message(PROMPT)], Some(chat_config(64, false)))
        .await
        .expect("AR turn 1 failed");
    assert_eq!(
        ar1.finish_reason, "stop",
        "fixture must stop early on EOS, got {:?} ({} tokens)",
        ar1.finish_reason, ar1.num_tokens
    );
    let ar2 = model
        .chat_session_continue(
            FOLLOW_UP.to_string(),
            None,
            None,
            Some(chat_config(64, false)),
        )
        .await
        .expect("AR turn 2 (continue) failed");

    // DSpark: same 2-turn shape on the same instance (the fresh start's
    // prefix verification resets the AR session's caches).
    let sp1 = model
        .chat_session_start(vec![user_message(PROMPT)], Some(chat_config(64, true)))
        .await
        .expect("DSpark turn 1 failed");
    assert_matches_ar("stop_turn1", &sp1, &ar1);

    let sp2 = model
        .chat_session_continue(
            FOLLOW_UP.to_string(),
            None,
            None,
            Some(chat_config(64, true)),
        )
        .await
        .expect("DSpark turn 2 (continue) failed");
    let perf2 = sp2.performance.as_ref().expect("turn 2 perf missing");
    assert!(
        perf2.mtp_cycles.unwrap_or(0) > 0,
        "the warm-continue turn must also run DSpark cycles"
    );
    assert!(
        sp2.cached_tokens > 0,
        "turn 2 must warm-continue on the saved session (cached_tokens > 0), got {}",
        sp2.cached_tokens
    );

    // 2-turn transcript parity vs the AR baseline.
    let ar_transcript = format!("{}\n---\n{}", ar1.text, ar2.text);
    let sp_transcript = format!("{}\n---\n{}", sp1.text, sp2.text);
    assert_eq!(
        sp_transcript, ar_transcript,
        "2-turn DSpark transcript diverged from the AR 2-turn baseline \
         (turn2 finish: dspark={:?} ar={:?})",
        sp2.finish_reason, ar2.finish_reason
    );
    assert_eq!(
        sp2.raw_text, ar2.raw_text,
        "warm-continue turn raw_text diverged from AR"
    );
    assert_eq!(sp2.finish_reason, ar2.finish_reason);
    println!(
        "[stop_then_delta] turn1={:?} ({} tok) turn2={:?} ({} tok, cached={})",
        sp1.finish_reason, sp1.num_tokens, sp2.finish_reason, sp2.num_tokens, sp2.cached_tokens
    );
}

// ---------------------------------------------------------------------------
// 4. Draft-fidelity oracle: acceptance sanity
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (real 12B + draft)"]
async fn dspark_acceptance_sanity() {
    let Some((model_path, draft_path)) = env_paths() else {
        return;
    };
    let model = load_dspark_model(&model_path, &draft_path).await;

    // Natural prose: the regime the draft was distilled on. matches-AR
    // cannot catch draft bugs (verification re-derives ground truth from
    // the target), they only DEPRESS acceptance — so gate on the headline
    // accept rate instead.
    const PROMPT: &str = "Explain in two paragraphs why the sky is blue during the day and \
         reddish at sunset.";
    let dspark = model
        .chat_session_start(vec![user_message(PROMPT)], Some(chat_config(200, true)))
        .await
        .expect("DSpark chat_session_start failed");
    let perf = dspark.performance.as_ref().expect("perf missing");
    let total = perf
        .mtp_mean_accepted_tokens_total
        .expect("mtp_mean_accepted_tokens_total missing after a DSpark run");
    println!(
        "[acceptance_sanity] mean_accepted_tokens_total={:.3} cycles={:?} mean_depth={:?} \
         decode_tok_s={:.1} ttft_ms={:.1}",
        total, perf.mtp_cycles, perf.mtp_mean_depth, perf.decode_tokens_per_second, perf.ttft_ms,
    );
    assert!(
        total >= 2.0,
        "draft-fidelity floor: expected mean accepted tokens/cycle >= 2.0 on natural prose, \
         got {total:.3} (a broken draft/context wiring depresses acceptance without breaking \
         matches-AR)"
    );
}

// ---------------------------------------------------------------------------
// 5. Sampled-temperature smoke
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (real 12B + draft)"]
async fn dspark_sampled_smoke() {
    let Some((model_path, draft_path)) = env_paths() else {
        return;
    };
    let model = load_dspark_model(&model_path, &draft_path).await;

    let mut cfg = chat_config(128, true);
    cfg.temperature = Some(0.7);
    let dspark = model
        .chat_session_start(
            vec![user_message(
                "Describe an imaginary small coastal town in a few sentences.",
            )],
            Some(cfg),
        )
        .await
        .expect("sampled DSpark chat_session_start failed");

    assert!(dspark.num_tokens > 0, "sampled run must emit tokens");
    assert!(
        !dspark.text.trim().is_empty(),
        "sampled run must produce text"
    );
    assert!(
        dspark.finish_reason == "stop" || dspark.finish_reason == "length",
        "unexpected finish_reason {:?}",
        dspark.finish_reason
    );
    let perf = dspark.performance.as_ref().expect("perf missing");
    let cycles = perf.mtp_cycles.unwrap_or(0);
    assert!(cycles > 0, "sampled run must still take the DSpark path");
    let total = perf
        .mtp_mean_accepted_tokens_total
        .expect("acceptance stats missing");
    assert!(
        (1.0..=8.0).contains(&total),
        "mean accepted tokens/cycle {total:.3} out of the sane [1, 1+block_size] range"
    );
    println!(
        "[sampled_smoke] T=0.7 tokens={} finish={:?} cycles={} mean_accepted_total={:.3}",
        dspark.num_tokens, dspark.finish_reason, cycles, total
    );
}

// ---------------------------------------------------------------------------
// 6. Streaming == sync
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (real 12B + draft)"]
async fn dspark_stream_matches_send() {
    let Some((model_path, draft_path)) = env_paths() else {
        return;
    };
    let model = load_dspark_model(&model_path, &draft_path).await;

    const PROMPT: &str = "List four common uses of a paperclip, one short sentence each.";

    let sync = model
        .chat_session_start(vec![user_message(PROMPT)], Some(chat_config(160, true)))
        .await
        .expect("sync DSpark chat_session_start failed");

    let (_handle, mut rx) = model
        .chat_stream_session_start_for_test(
            vec![user_message(PROMPT)],
            Some(chat_config(160, true)),
        )
        .expect("stream dispatch failed");

    let mut visible = String::new();
    let mut done_chunk = None;
    while let Some(result) = rx.recv().await {
        let chunk = result.expect("stream chunk error");
        if chunk.done {
            done_chunk = Some(chunk);
            break;
        }
        if chunk.is_reasoning != Some(true) {
            visible.push_str(&chunk.text);
        }
    }
    let done = done_chunk.expect("stream ended without a terminal done-chunk");

    // The strongest stream-vs-sync oracle: the terminal chunk's raw_text is
    // the full undecoded generation — byte-equality pins that the streaming
    // turn committed the exact same token sequence as the sync turn.
    assert_eq!(
        done.raw_text.as_deref(),
        Some(sync.raw_text.as_str()),
        "streaming DSpark turn generated a different token stream than the sync turn"
    );
    // Chunk-routing parity, modulo the parser's TRAILING trim: on a
    // channel-only output the sync result promotes `parser.thinking()`
    // (which `.trim()`s — `output_parser.rs`) into `.text`, while the
    // streaming promotion emits the buffered reasoning untrimmed — a
    // PRE-EXISTING gemma4 emitter/parser semantic shared with the plain AR
    // path (verified: an AR stream-vs-sync run of this same prompt shows
    // the identical trailing-whitespace delta with byte-equal raw_text).
    assert_eq!(
        visible.trim_end(),
        sync.text.trim_end(),
        "concatenated streaming chunks diverged from the sync DSpark text"
    );
    assert_eq!(
        done.finish_reason.as_deref(),
        Some(sync.finish_reason.as_str()),
        "terminal chunk finish_reason mismatch"
    );
    assert_eq!(done.num_tokens, Some(sync.num_tokens));
    let perf = done
        .performance
        .as_ref()
        .expect("terminal chunk must carry performance");
    assert!(
        perf.mtp_cycles.unwrap_or(0) > 0,
        "the streaming turn must run DSpark cycles"
    );
    println!(
        "[stream_matches_send] tokens={:?} finish={:?} cycles={:?}",
        done.num_tokens, done.finish_reason, perf.mtp_cycles
    );
}
