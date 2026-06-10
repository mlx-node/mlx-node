//! Gated e2e smoke test for the sym8 (per-output-channel symmetric int8)
//! Qwen3.5 dense path.
//!
//! Loads a sym8-converted checkpoint through the production loader (which
//! must build the int8 W8A8 `QuantizedLinear`s, skip the C++ compiled-forward
//! registration, and run the EAGER Rust forward) and runs one short greedy
//! turn. The gate is COHERENT text — a mis-dispatched sym8 layer (e.g. the
//! compiled path's no-biases→MXFP8 heuristic) produces `!!!!`/garbage
//! immediately.
//!
//! Run manually (needs an M5+ GPU — the loader fail-louds on gen < 17):
//!
//! ```shell
//! MLX_SYM8_TEST_MODEL_PATH=/tmp/qwen35-0.8b-sym8-mlx \
//!     cargo test -p mlx-core --release --test qwen3_5_sym8_smoke -- --ignored --nocapture
//! ```
//!
//! Optional: `MLX_SYM8_DEBUG=1` prints per-forward kernel dispatch lines
//! (`[sym8] qmv M=1 ...` during decode, `[sym8] gemm M=<prefill> ...`).

use std::path::Path;

use mlx_core::models::qwen3_5::model::{ChatConfig, Qwen3_5Model};
use mlx_core::tokenizer::ChatMessage;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_SYM8_TEST_MODEL_PATH pointing to a sym8-converted Qwen3.5 dense checkpoint"]
async fn sym8_checkpoint_loads_and_generates_coherent_text() {
    let Ok(model_path) = std::env::var("MLX_SYM8_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_SYM8_TEST_MODEL_PATH unset (point it at e.g. /tmp/qwen35-0.8b-sym8-mlx)"
        );
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_SYM8_TEST_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load sym8 Qwen3.5 model");

    let cfg = ChatConfig {
        max_new_tokens: Some(160),
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
        thinking_token_budget: Some(48),
        include_reasoning: Some(true),
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: None,
        mtp_depth: None,
        mtp_adaptive_depth: None,
    };
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "What is the capital of France? Answer in one short sentence.".to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        images: None,
    }];

    let result = model
        .chat_session_start(messages, Some(cfg))
        .await
        .expect("sym8 greedy chat turn failed");

    let text = result.text.trim().to_string();
    println!("=== sym8 greedy output ({} tokens) ===", result.num_tokens);
    println!("{}", text);
    println!("=== end ===");

    assert!(!text.is_empty(), "sym8 generation produced empty text");
    // Garbage detectors: a broken int8 dispatch reliably emits long runs of a
    // single repeated character (classically `!!!!`) or the replacement char.
    assert!(
        !text.contains("!!!!"),
        "sym8 generation looks like garbage (repeated '!'): {text}"
    );
    assert!(
        !text.contains('\u{fffd}'),
        "sym8 generation contains replacement characters: {text}"
    );
    let distinct = {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        chars.len()
    };
    assert!(
        distinct >= 10,
        "sym8 generation has only {distinct} distinct characters — likely garbage: {text}"
    );
}
