//! Real-LFM2 continuous-batching token parity and occupancy gate.
//!
//! The serial oracle deliberately runs through the scheduler one request at a
//! time. Its decode therefore carries the incremental ShortConv state produced
//! by each preceding token. It does not use the legacy Pass-1 reconstruction
//! path as a numerical oracle (that path differs by roughly 40 ULP on bf16
//! checkpoints and can flip near-tied greedy tokens).
//!
//! Run with:
//! `MLX_TEST_LFM2_MODEL_PATH=/abs/lfm2 cargo test -p mlx-core --test lfm2_concurrent_batched_parity -- --ignored --nocapture`

use std::path::PathBuf;

use futures::future::join_all;
use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::lfm2::model::Lfm2Model;
use mlx_core::tokenizer::ChatMessage;

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("MLX_TEST_LFM2_MODEL_PATH")?;
    let path = PathBuf::from(path);
    assert!(path.exists(), "MLX_TEST_LFM2_MODEL_PATH does not exist");
    Some(path)
}

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

fn config(owner: &str) -> ChatConfig {
    ChatConfig {
        cache_salt: None,
        cache_owner_id: Some(owner.to_string()),
        cache_root_owner_id: Some(owner.to_string()),
        max_new_tokens: Some(24),
        temperature: Some(0.0),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        max_consecutive_tokens: Some(0),
        max_ngram_repeats: Some(0),
        ngram_size: Some(0),
        thinking_token_budget: Some(12),
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(true),
        ..ChatConfig::default()
    }
}

fn assert_same(expected: &ChatResult, actual: &ChatResult, prompt: &str) {
    assert_eq!(actual.text, expected.text, "text mismatch for {prompt:?}");
    assert_eq!(
        actual.raw_text, expected.raw_text,
        "raw_text mismatch for {prompt:?}"
    );
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert_eq!(actual.num_tokens, expected.num_tokens);
    assert_eq!(actual.prompt_tokens, expected.prompt_tokens);
    assert_eq!(
        actual.cached_tokens, expected.cached_tokens,
        "prefix accounting mismatch for {prompt:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn carried_state_serial_and_n2_batch_are_token_identical() {
    let Some(path) = model_path() else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model = Lfm2Model::load_from_dir(&path.to_string_lossy())
        .await
        .expect("load lfm2");
    assert!(model.has_block_paged_cache(), "gate requires paged LFM2");
    assert!(
        model.max_concurrent_sequences() >= 2,
        "LFM2 scheduler must advertise a batched lane"
    );

    let prompts = [
        "Give a concise explanation of why leaves look green.",
        "Give a concise explanation of why ocean tides change.",
    ];
    let mut serial = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let result = model
            .chat_session_start(
                vec![user_message(prompt)],
                Some(config(&format!("serial-{index}"))),
            )
            .await
            .expect("carried-state serial turn");
        assert_eq!(result.cached_tokens, 0, "serial oracle must be cold");
        serial.push(result);
        model.reset_caches().await.expect("reset serial oracle");
    }

    let batched = join_all(prompts.iter().enumerate().map(|(index, prompt)| {
        model.chat_session_start(
            vec![user_message(prompt)],
            Some(config(&format!("batch-{index}"))),
        )
    }))
    .await;
    for ((expected, actual), prompt) in serial.iter().zip(batched).zip(prompts) {
        assert_same(expected, &actual.expect("batched LFM2 turn"), prompt);
    }

    let stats = model.scheduler_stats().await.expect("scheduler stats");
    assert!(
        stats.max_batch_occupancy >= 2,
        "expected a real LFM2 N=2 decode step, max={} hist={:?}",
        stats.max_batch_occupancy,
        stats
            .decode_batch_occupancy_hist
            .iter()
            .map(|bucket| (bucket.occupancy, bucket.steps))
            .collect::<Vec<_>>()
    );
    assert!(stats.block_capacity > 0, "live block telemetry is required");
    assert_eq!(stats.rows_alloc_evicted, 0.0);
}
