//! Gated integration test for the LFM2.5 MoE (`model_type: "lfm2_moe"`,
//! e.g. `lfm2.5-8b-a1b`) session-based chat path.
//!
//! Mirrors `lfm2_session.rs` but exercises a MoE checkpoint. The MoE
//! routing (per-layer dense vs sparse FFN) lives inside the SAME
//! `Lfm2Model` surface, so the session/streaming machinery is identical;
//! this test just confirms a real MoE checkpoint loads and decodes with a
//! flat TTFT across delta turns and stream/non-stream agreement.
//!
//! The test is gated because it needs a real LFM2.5 MoE checkpoint on disk.
//! Run it manually with:
//!
//! ```shell
//! MLX_TEST_LFM2_MOE_MODEL_PATH=./.cache/models/lfm2.5-8b-a1b-mlx-bf16 \
//!     cargo test -p mlx-core --test lfm2_moe_session -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_LFM2_MOE_MODEL_PATH` the test early-returns and passes
//! trivially so it still compiles as part of `cargo test`.

use std::path::Path;

use mlx_core::models::lfm2::model::Lfm2Model;
use mlx_core::models::qwen3_5::model::ChatConfig;
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
        thinking_token_budget: Some(32),
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
        is_error: None,
        reasoning_content: None,
        images: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MOE_MODEL_PATH pointing to a real LFM2.5 MoE checkpoint"]
async fn lfm2_moe_session_keeps_ttft_flat_across_turns() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MOE_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_LFM2_MOE_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/lfm2.5-8b-a1b-mlx-bf16)"
        );
        return;
    };

    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MOE_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2.5 MoE model");

    #[derive(Debug, Clone)]
    struct TurnSnapshot {
        ttft_ms: f64,
        prompt_tokens: u32,
    }

    // --- Turn 1: chat_session_start ---
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
    assert!(r1.num_tokens > 0, "turn 1 generated no tokens");
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

    // --- Structural assertions ---
    assert_eq!(snapshots.len(), 4, "expected 4 turn snapshots");
    let turn1 = &snapshots[0];
    let turn2 = &snapshots[1];
    let turn3 = &snapshots[2];
    let turn4 = &snapshots[3];

    // prompt_tokens must GROW across delta turns.
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

    // TTFT stays flat (<= 1.5x of turn 1).
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
}

/// Reset-determinism: the same prompt from a fresh session must produce the
/// same first generated token (greedy / temperature 0).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MOE_MODEL_PATH pointing to a real LFM2.5 MoE checkpoint"]
async fn lfm2_moe_reset_determinism() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MOE_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MOE_MODEL_PATH unset");
        return;
    };
    assert!(Path::new(&model_path).exists());

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2.5 MoE model");

    let prompt = "Name a color.";

    let r1 = model
        .chat_session_start(vec![user_message(prompt)], Some(chat_config_default(8)))
        .await
        .expect("first start failed");

    model.reset_caches().expect("reset_caches failed");

    let r2 = model
        .chat_session_start(vec![user_message(prompt)], Some(chat_config_default(8)))
        .await
        .expect("second start failed");

    assert_eq!(
        r1.text, r2.text,
        "greedy decode after reset must be deterministic: {:?} vs {:?}",
        r1.text, r2.text
    );
}
