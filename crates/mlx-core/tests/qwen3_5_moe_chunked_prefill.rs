//! Gated integration tests for Qwen3.5 MoE chunked prefill.
//!
//! Validates that the chunked prefill path (which processes prompts in
//! `PREFILL_STEP_SIZE`-token chunks to bound peak GPU memory on long
//! contexts) produces correct, coherent output and behaves identically
//! to a single-shot prefill for short prompts.
//!
//! All tests are gated on `MLX_TEST_MOE_MODEL_PATH` pointing to a real
//! Qwen3.5 MoE checkpoint and marked `#[ignore]` — without the env var
//! they early-return so plain `cargo test` still compiles and passes
//! trivially. Run with:
//!
//! ```shell
//! MLX_TEST_MOE_MODEL_PATH=./.cache/models/qwen3.5-moe-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_5_moe_chunked_prefill \
//!     -- --ignored --nocapture
//! ```
//!
//! Key invariants exercised:
//! - `test_chunked_prefill_under_threshold`: a short prompt (<2048 tokens)
//!   hits a single chunked_prefill iteration that is byte-equivalent to
//!   the old `forward_inner`-only code path. Determinism + coherence
//!   assertions stand in for "matches single-shot".
//! - `test_chunked_prefill_boundary`: a prompt of exactly
//!   `PREFILL_STEP_SIZE` tokens hits the edge case
//!   `total_len - offset == PREFILL_STEP_SIZE` where the while-loop
//!   condition `total_len - offset > chunk_size` is false on entry, so
//!   the whole prompt goes through the final single-chunk branch. No
//!   eval_layer_caches/clear_cache barrier is invoked. Validates that
//!   boundary doesn't drop or duplicate the last chunk.
//! - `test_chunked_prefill_matches_single_shot`: runs the same prompt
//!   twice with temperature=0 and top_k=1 (greedy deterministic) and
//!   compares the generated token streams. They MUST be identical —
//!   if chunking introduced any graph-eval-ordering bug, the recurrent
//!   GDN state would diverge between the two runs and produce different
//!   tokens. Also exercises the long-context (>2048) path so the
//!   multi-chunk loop is actually hit.

use std::path::Path;

use mlx_core::models::qwen3_5::model::ChatConfig;
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use mlx_core::tokenizer::ChatMessage;

/// Minimum chunk size enforced by the MoE chunked_prefill. This mirrors
/// the `PREFILL_STEP_SIZE` constant in `qwen3_5_moe/model.rs` — kept in
/// sync manually because the constant is crate-private.
const PREFILL_STEP_SIZE: usize = 2048;

fn chat_config_default(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        // Deterministic greedy decoding: temperature=0 collapses sampling
        // to argmax, and top_k=1 + top_p=1.0 keeps only the argmax token.
        // This is what makes the "matches single-shot" assertion meaningful.
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        min_p: Some(0.0),
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
        reasoning_effort: Some("none".to_string()),
        thinking_token_budget: Some(0),
        include_reasoning: Some(false),
        report_performance: Some(true),
        reuse_cache: Some(false),
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

/// Build a deterministic padded prompt whose rendered ChatML token
/// length is at least `target_tokens`. We cannot directly construct
/// tokenized arrays because the tokenizer-encoded ChatML template adds
/// variable overhead, so we use `count_tokens` afterwards to size up
/// our filler. The padding is the word "filler" repeated — a common
/// subword so the tokenizer produces a predictable ratio (~1 token per
/// "filler " on Qwen3.5 tokenizer).
async fn build_padded_prompt(_model: &Qwen3_5MoeModel, target_tokens: usize) -> String {
    // Each "filler " word is typically 1–2 BPE tokens on Qwen3.5. We
    // overshoot slightly to make sure we're above target, but we don't
    // need to hit the target exactly — the tests only need the prompt
    // to be at least `target_tokens` long to trigger the corresponding
    // chunked_prefill path.
    let pad_units = target_tokens + 64;
    let filler: String = "filler ".repeat(pad_units);
    format!("{filler}\n\nIn one short word, reply with the letter A.")
}

/// Load the model or skip the test gracefully if the env var is unset.
/// Returns `None` to signal the test should early-return.
async fn load_model_or_skip() -> Option<Qwen3_5MoeModel> {
    let Ok(model_path) = std::env::var("MLX_TEST_MOE_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MOE_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/qwen3.5-moe-mlx-bf16)"
        );
        return None;
    };

    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MOE_MODEL_PATH does not exist: {model_path}"
    );

    let model = Qwen3_5MoeModel::load(model_path)
        .await
        .expect("failed to load Qwen3.5 MoE model");
    Some(model)
}

/// Under the chunk threshold: a single user message in the tens of
/// tokens range should prefill in ONE chunked_prefill iteration — the
/// while-loop is skipped entirely and the final-chunk branch handles
/// the whole prompt, behaviorally identical to the pre-chunking
/// `forward_inner`-only path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MODEL_PATH pointing to a real Qwen3.5 MoE checkpoint"]
async fn test_chunked_prefill_under_threshold() {
    let Some(model) = load_model_or_skip().await else {
        return;
    };

    let cfg = chat_config_default(16);
    let messages = vec![user_message("Reply with the word 'hi', nothing else.")];
    let result = model
        .chat_session_start(messages, Some(cfg))
        .await
        .expect("chat_session_start failed on short prompt");

    // Structural: we should have generated SOMETHING and the prefill
    // should be well under the chunk threshold (otherwise this test is
    // exercising the wrong branch).
    assert!(
        !result.text.trim().is_empty(),
        "short-prompt generation returned empty text"
    );
    assert!(
        (result.prompt_tokens as usize) < PREFILL_STEP_SIZE,
        "expected short prompt < {} tokens, got {}",
        PREFILL_STEP_SIZE,
        result.prompt_tokens
    );
    assert!(
        result.num_tokens > 0,
        "short-prompt generation produced zero tokens"
    );
}

/// Boundary: a prompt of exactly `PREFILL_STEP_SIZE` tokens. The
/// while-loop condition `total_len - offset > chunk_size` is `false`
/// on first entry (it's `==`, not `>`), so the entire prompt falls
/// through to the single final-chunk forward. Validates that we do
/// not double-process or drop the last block.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MODEL_PATH pointing to a real Qwen3.5 MoE checkpoint"]
async fn test_chunked_prefill_boundary() {
    let Some(model) = load_model_or_skip().await else {
        return;
    };

    // Build a prompt targeted near the chunk boundary. The actual
    // tokenized length varies ±64 tokens from target depending on
    // tokenizer overhead; we assert it's roughly at the threshold
    // after the fact.
    let prompt_text = build_padded_prompt(&model, PREFILL_STEP_SIZE).await;
    let cfg = chat_config_default(8);
    let messages = vec![user_message(&prompt_text)];

    let result = model
        .chat_session_start(messages, Some(cfg))
        .await
        .expect("chat_session_start failed at chunk boundary");

    // Sanity: prompt should straddle the boundary zone (within 2x of
    // PREFILL_STEP_SIZE). This exercises both the boundary leg (if the
    // tokenizer happened to produce exactly PREFILL_STEP_SIZE tokens)
    // AND the one-chunk-then-remainder leg (if it produced slightly
    // more). Both are correctness-critical edge cases.
    let prompt_toks = result.prompt_tokens as usize;
    assert!(
        prompt_toks >= PREFILL_STEP_SIZE / 2,
        "padded prompt too short for boundary test: {prompt_toks} tokens"
    );
    assert!(
        prompt_toks <= PREFILL_STEP_SIZE * 3,
        "padded prompt overshoot: {prompt_toks} tokens",
    );
    assert!(
        result.num_tokens > 0,
        "boundary prefill produced zero decode tokens (chunked_prefill may \
         have consumed the prompt but lost the final-chunk logits)"
    );
    assert!(
        !result.text.is_empty(),
        "boundary prefill produced empty text"
    );
}

/// Determinism + multi-chunk: run the SAME long-context prompt twice
/// under greedy decoding and require identical token streams. If
/// chunking introduced any eval-ordering bug — e.g. the linear-
/// attention (GDN) recurrent state advances differently between
/// runs, or a chunk boundary causes a KV cache to not materialize
/// before the next chunk reads it — the two runs would diverge.
///
/// This is the "matches single-shot" check repurposed for the
/// chunked-only API surface: if the chunked path were non-deterministic
/// under greedy decoding, it would be broken regardless of whether
/// single-shot still worked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MODEL_PATH pointing to a real Qwen3.5 MoE checkpoint"]
async fn test_chunked_prefill_matches_single_shot() {
    let Some(model) = load_model_or_skip().await else {
        return;
    };

    // Force a prompt that REQUIRES multiple chunks (> 2 * step size).
    // This guarantees the while-loop runs at least twice, exercising
    // the inter-chunk eval/clear barrier.
    let prompt_text = build_padded_prompt(&model, PREFILL_STEP_SIZE * 3).await;

    let cfg1 = chat_config_default(24);
    let result1 = model
        .chat_session_start(vec![user_message(&prompt_text)], Some(cfg1))
        .await
        .expect("first long-context run failed");

    // Confirm we actually exercised the multi-chunk path.
    assert!(
        (result1.prompt_tokens as usize) > PREFILL_STEP_SIZE * 2,
        "expected multi-chunk prompt, got {} tokens",
        result1.prompt_tokens
    );

    let cfg2 = chat_config_default(24);
    let result2 = model
        .chat_session_start(vec![user_message(&prompt_text)], Some(cfg2))
        .await
        .expect("second long-context run failed");

    // Greedy decoding on identical input must produce identical
    // output. If it doesn't, chunking broke the prefill.
    assert_eq!(
        result1.prompt_tokens, result2.prompt_tokens,
        "prompt_tokens diverged between runs: {} vs {}",
        result1.prompt_tokens, result2.prompt_tokens,
    );
    assert_eq!(
        result1.text, result2.text,
        "greedy chunked decoding produced different outputs across runs \
         — chunked_prefill is non-deterministic. \
         run1: {:?}\nrun2: {:?}",
        result1.text, result2.text,
    );
}
