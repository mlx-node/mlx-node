//! Real-checkpoint smoke for dense Qwen3.8 plus the external DFlash2 drafter.
//!
//! ```shell
//! MLX_TEST_QWEN38_TARGET_PATH=/path/to/qwen3.8-27b-mlx \
//! MLX_TEST_QWEN38_DFLASH2_PATH=/path/to/qwen3.8-27b-dflash2 \
//! cargo test -p mlx-core --test qwen3_8_dflash2 -- --ignored --nocapture
//! ```

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::{Qwen3_5Model, Qwen35LoadOptions};
use mlx_core::tokenizer::ChatMessage;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires real Qwen3.8 target and DFlash2 checkpoints"]
async fn qwen38_dflash2_loads_and_runs_a_verified_cycle() {
    let target =
        std::env::var("MLX_TEST_QWEN38_TARGET_PATH").expect("set MLX_TEST_QWEN38_TARGET_PATH");
    let draft =
        std::env::var("MLX_TEST_QWEN38_DFLASH2_PATH").expect("set MLX_TEST_QWEN38_DFLASH2_PATH");
    let model = Qwen3_5Model::load(
        target,
        Some(Qwen35LoadOptions {
            draft_model_path: Some(draft),
        }),
    )
    .await
    .expect("load Qwen3.8 target with DFlash2 companion");
    assert!(
        model.has_mtp_weights(),
        "external DFlash2 must advertise speculative capability"
    );

    let result = model
        .chat_session_start(
            vec![user_message(
                "Reply with one short sentence about the Moon.",
            )],
            Some(ChatConfig {
                max_new_tokens: Some(8),
                temperature: Some(0.0),
                report_performance: Some(true),
                enable_mtp: Some(true),
                mtp_depth: Some(2),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            }),
        )
        .await
        .expect("DFlash2 chat turn");
    assert!(result.num_tokens > 0 && result.num_tokens <= 8);
    assert!(
        result
            .performance
            .as_ref()
            .and_then(|performance| performance.mtp_cycles)
            .is_some_and(|cycles| cycles > 0),
        "smoke must execute at least one DFlash2 propose/verify cycle"
    );

    // Exercise both cache owners without an explicit reset. Ordinary AR uses
    // the target's default-on paged adapter; returning to DFlash2 must cross a
    // cold flat-lane barrier and reproduce the original greedy result.
    let paged = model
        .chat_session_start(
            vec![user_message("Reply with one short sentence about the Sun.")],
            Some(ChatConfig {
                max_new_tokens: Some(5),
                temperature: Some(0.0),
                report_performance: Some(true),
                enable_mtp: Some(false),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            }),
        )
        .await
        .expect("paged AR turn between DFlash2 turns");
    assert!(paged.num_tokens > 0 && paged.num_tokens <= 5);
    assert!(
        paged
            .performance
            .as_ref()
            .is_some_and(|performance| performance.mtp_cycles.is_none()),
        "enable_mtp=false must stay on plain paged AR"
    );

    let replay = model
        .chat_session_start(
            vec![user_message(
                "Reply with one short sentence about the Moon.",
            )],
            Some(ChatConfig {
                max_new_tokens: Some(8),
                temperature: Some(0.0),
                report_performance: Some(true),
                enable_mtp: Some(true),
                mtp_depth: Some(2),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            }),
        )
        .await
        .expect("DFlash2 turn after paged AR");
    assert_eq!(
        replay.raw_text, result.raw_text,
        "paged-to-DFlash2 transition must match a cold greedy DFlash2 turn"
    );
    assert_eq!(
        replay.cached_tokens, 0,
        "paged target K/V must not be reported as reusable flat DFlash2 state"
    );
    assert!(
        replay
            .performance
            .as_ref()
            .and_then(|performance| performance.mtp_cycles)
            .is_some_and(|cycles| cycles > 0),
        "DFlash2 must resume after crossing the paged cache barrier"
    );

    model
        .reset_caches()
        .await
        .expect("reset before sampled DFlash2 turn");
    let sampled = model
        .chat_session_start(
            vec![user_message("Name one color.")],
            Some(ChatConfig {
                max_new_tokens: Some(5),
                temperature: Some(0.7),
                top_p: Some(0.9),
                report_performance: Some(true),
                enable_mtp: Some(true),
                mtp_depth: Some(2),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            }),
        )
        .await
        .expect("sampled sparse-selector DFlash2 turn");
    assert!(sampled.num_tokens > 0 && sampled.num_tokens <= 5);
    assert!(
        sampled
            .performance
            .as_ref()
            .and_then(|performance| performance.mtp_cycles)
            .is_some_and(|cycles| cycles > 0),
        "sampled smoke must execute sparse proposal correction"
    );

    model
        .shutdown_for_test()
        .expect("Qwen3.8 model thread must shut down cleanly");
}
