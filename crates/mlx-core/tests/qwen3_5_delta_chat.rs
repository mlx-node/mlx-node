//! Gated integration test for the Phase 1 session-based chat delta path.
//!
//! This test exercises `Qwen3_5Model::chat_tokens_delta_blocking`, the
//! non-NAPI bridge that dispatches `Qwen35Cmd::ChatTokensDelta` to the
//! dedicated model thread. It validates that TTFT stays roughly flat
//! across 4 turns — proving the KV caches are actually being reused and
//! the new turn only pays for its delta prefill, not a full re-prefill
//! of the accumulating history.
//!
//! The test is gated because it needs a real Qwen3.5 Dense checkpoint
//! on disk. Run it manually with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/qwen3.5-0.8b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_5_delta_chat -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_MODEL_PATH` the test early-returns and passes
//! trivially so it still compiles as part of `cargo test`.

use std::path::Path;

use mlx_core::models::qwen3_5::model::{ChatConfig, Qwen3_5Model};
use mlx_core::tokenizer::{ChatMessage, Qwen3Tokenizer};

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

/// Tokenize a literal assistant-reply-then-new-user-turn delta string,
/// the same way the TS session class will in Phase 2.
///
/// Callers are responsible for the exact wire format — this test just
/// verifies that the delta path does not re-prefill the entire conversation.
fn encode_delta(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    let encoding = tokenizer
        .encode(text, false)
        .expect("encoding delta text failed");
    encoding.get_ids().to_vec()
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn delta_path_keeps_ttft_flat_across_turns() {
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

    // Load a companion tokenizer directly so the test can build raw
    // delta token sequences without going through the jinja template.
    let tokenizer_json = model_dir.join("tokenizer.json");
    assert!(
        tokenizer_json.exists(),
        "expected tokenizer.json at {}",
        tokenizer_json.display()
    );
    let raw_tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_json)
        .expect("failed to load raw tokenizers::Tokenizer");
    let qwen_tokenizer =
        Qwen3Tokenizer::from_file(&tokenizer_json).expect("failed to load Qwen3Tokenizer");
    let im_end_id = qwen_tokenizer
        .im_end_id()
        .expect("tokenizer missing <|im_end|>");

    // Load the model via the normal async path.
    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");

    // --- Turn 1: regular chat_sync path (establishes the session) ---
    let turn1_cfg = chat_config_default(64);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let r1 = model
        .chat(turn1_messages, Some(turn1_cfg))
        .await
        .expect("turn 1 chat failed");
    let ttft1 = r1
        .performance
        .as_ref()
        .expect("turn 1 performance missing")
        .ttft_ms;
    println!(
        "turn 1 ttft={:.1}ms prompt_tokens={} num_tokens={}",
        ttft1, r1.prompt_tokens, r1.num_tokens
    );

    // --- Turns 2..=4: delta path ---
    //
    // We build a delta of the form
    //   <assistant-reply>\n<|im_end|>\n<|im_start|>user\n<next>\n<|im_end|>\n<|im_start|>assistant\n
    // where <assistant-reply> is the raw_text from the previous turn.
    //
    // The cached token history at this point is the full Turn-1 prompt
    // (post-chat_sync `save_cache_state_direct` appends the generated
    // tokens too, trimming the trailing token when finish_reason=="length"),
    // so the delta just has to close the assistant turn and open a new one.
    let mut prev_raw = r1.raw_text.clone();
    let user_followups = [
        "And in another word?",
        "Any synonym?",
        "One more, different?",
    ];
    let mut ttfts = Vec::new();
    ttfts.push(ttft1);

    for (idx, next_user) in user_followups.iter().enumerate() {
        let turn_idx = idx + 2;
        // Close the last assistant turn with <|im_end|>, then open the
        // new user turn. We intentionally mirror the jinja template's
        // raw format for Qwen3.5 ChatML.
        let delta_text = format!(
            "{assistant_tail}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
            assistant_tail = prev_raw,
            user = next_user,
        );
        let delta_tokens = encode_delta(&raw_tokenizer, &delta_text);
        assert!(!delta_tokens.is_empty());

        let cfg = chat_config_default(64);
        let result = model
            .chat_tokens_delta_blocking(delta_tokens.clone(), cfg)
            .expect("delta chat failed");
        let ttft = result
            .performance
            .as_ref()
            .expect("delta performance missing")
            .ttft_ms;
        println!(
            "turn {turn_idx} ttft={:.1}ms prompt_tokens={} num_tokens={} delta_len={}",
            ttft,
            result.prompt_tokens,
            result.num_tokens,
            delta_tokens.len()
        );

        // Cache must have been extended: prompt_tokens should grow roughly
        // linearly (cached history + delta), but TTFT should NOT.
        assert!(result.prompt_tokens >= delta_tokens.len() as u32);

        ttfts.push(ttft);
        // The raw reply from chat_tokens_delta ends before <|im_end|>
        // because the decode loop stops on im_end_id. Chain on it.
        prev_raw = result.raw_text.clone();

        // Sanity: the session path uses <|im_end|> as its stop token,
        // so a successful turn must NOT produce the id in the generated
        // sequence (it stops exactly on it). We can't directly inspect
        // generated_tokens here, but finish_reason carries the signal.
        assert!(
            result.finish_reason == "stop" || result.finish_reason == "length",
            "unexpected finish_reason: {}",
            result.finish_reason
        );
        let _ = im_end_id; // (keep the binding referenced for clarity)
    }

    // Core assertion: TTFT stays flat (≤1.5x of turn 1) across all turns.
    // The broken pre-Phase-1 path would balloon linearly as the history
    // grows — we pick 1.5x as a generous bound that still catches a
    // full re-prefill regression.
    let ttft_turn4 = *ttfts.last().unwrap();
    let bound = ttft1 * 1.5;
    assert!(
        ttft_turn4 < bound,
        "delta-path TTFT regression: turn1={:.1}ms turn4={:.1}ms bound={:.1}ms. \
         TTFTs: {:?}",
        ttft1,
        ttft_turn4,
        bound,
        ttfts
    );
}
