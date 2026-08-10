//! Real-Qwen3 continuous-batching parity and occupancy gate.
//!
//! Run with:
//! `QWEN3_STAGE1_MODEL_PATH=/abs/qwen3-0.6b-mlx-bf16 cargo test -p mlx-core --test concurrent_batched_parity -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use futures::future::join_all;
use mlx_core::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use mlx_core::models::qwen3::persistence::load_with_thread;
use mlx_core::tokenizer::ChatMessage;

struct ForceSerialGuard;

impl ForceSerialGuard {
    fn enable() -> Self {
        // This ignored integration test is the only test in its process. Rust
        // 2024 marks process-environment mutation unsafe because sibling
        // threads could read concurrently; the model thread is created only
        // after this write and every serial turn is drained before removal.
        unsafe { std::env::set_var("MLX_SERVE_FORCE_SERIAL", "1") };
        Self
    }
}

impl Drop for ForceSerialGuard {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("MLX_SERVE_FORCE_SERIAL") };
    }
}

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("QWEN3_STAGE1_MODEL_PATH")?;
    let path = PathBuf::from(path);
    assert!(path.exists(), "QWEN3_STAGE1_MODEL_PATH does not exist");
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
        cache_owner_id: Some(owner.to_string()),
        cache_root_owner_id: Some(owner.to_string()),
        max_new_tokens: Some(16),
        temperature: Some(0.0),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        max_consecutive_tokens: Some(0),
        max_ngram_repeats: Some(0),
        ngram_size: Some(0),
        thinking_token_budget: Some(8),
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
}

async fn drain_stream(
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<napi::Result<ChatStreamChunk>>,
) -> ChatStreamChunk {
    while let Some(chunk) = receiver.recv().await {
        let chunk = chunk.expect("streaming scheduler row failed");
        if chunk.done {
            return chunk;
        }
    }
    panic!("stream closed without a terminal chunk")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs QWEN3_STAGE1_MODEL_PATH pointing to a real Qwen3 checkpoint"]
async fn serial_uniform_batch_and_interleaved_streams_are_token_identical() {
    let Some(path) = model_path() else {
        eprintln!("skipping: QWEN3_STAGE1_MODEL_PATH unset");
        return;
    };
    let serial_guard = ForceSerialGuard::enable();
    let model = load_with_thread(&path.to_string_lossy())
        .await
        .expect("load qwen3");
    assert!(model.has_block_paged_cache(), "gate requires paged Qwen3");

    let prompts = [
        "Answer with one word: the color of grass is",
        "Answer with just the number: 7 plus 5 is",
        "Complete briefly: water freezes at",
        "Name one primary color.",
        "Answer with one word: the opposite of cold is",
        "Answer with just the number: 9 minus 4 is",
        "Complete briefly: birds can",
        "Name one common fruit.",
    ];
    let mut serial = Vec::new();
    let serial_started = Instant::now();
    for (index, prompt) in prompts.iter().enumerate() {
        serial.push(
            model
                .chat_session_start(
                    vec![user_message(prompt)],
                    Some(config(&format!("serial-{index}"))),
                )
                .await
                .expect("serial oracle turn"),
        );
    }
    let serial_elapsed = serial_started.elapsed();
    drop(serial_guard);
    model.reset_caches().await.expect("reset before batch");

    let batched_started = Instant::now();
    let batched = join_all(prompts.iter().enumerate().map(|(index, prompt)| {
        model.chat_session_start(
            vec![user_message(prompt)],
            Some(config(&format!("batch-{index}"))),
        )
    }))
    .await;
    let batched_elapsed = batched_started.elapsed();
    for ((expected, actual), prompt) in serial.iter().zip(batched).zip(prompts) {
        assert_same(expected, &actual.expect("batched turn"), prompt);
    }
    let stats = model.scheduler_stats().await.expect("scheduler stats");
    assert!(
        stats.max_batch_occupancy >= 8,
        "expected a real N=8 decode step, stats max={} hist={:?}",
        stats.max_batch_occupancy,
        stats
            .decode_batch_occupancy_hist
            .iter()
            .map(|bucket| (bucket.occupancy, bucket.steps))
            .collect::<Vec<_>>()
    );
    assert!(stats.block_capacity > 0, "live block telemetry is required");
    assert_eq!(stats.rows_alloc_evicted, 0.0);
    assert_eq!(stats.reserved_blocks, 0);
    eprintln!(
        "stage1 N=8 host+device wall: serial={:.2}ms batch={:.2}ms aggregate_speedup={:.2}x hist={:?}",
        serial_elapsed.as_secs_f64() * 1_000.0,
        batched_elapsed.as_secs_f64() * 1_000.0,
        serial_elapsed.as_secs_f64() / batched_elapsed.as_secs_f64(),
        stats
            .decode_batch_occupancy_hist
            .iter()
            .map(|bucket| (bucket.occupancy, bucket.steps))
            .collect::<Vec<_>>()
    );

    model.reset_caches().await.expect("reset before streams");
    let mut receivers = Vec::new();
    for (index, prompt) in prompts[..2].iter().enumerate() {
        let (_handle, receiver) = model
            .chat_stream_session_start_for_test(
                vec![user_message(prompt)],
                Some(config(&format!("stream-{index}"))),
            )
            .expect("dispatch stream");
        receivers.push(receiver);
    }
    let terminals = join_all(receivers.into_iter().map(drain_stream)).await;
    for ((expected, terminal), prompt) in serial.iter().zip(terminals).zip(prompts) {
        assert_eq!(
            terminal.raw_text.as_deref(),
            Some(expected.raw_text.as_str()),
            "stream raw_text mismatch for {prompt:?}"
        );
        assert_eq!(
            terminal.finish_reason.as_deref(),
            Some(expected.finish_reason.as_str())
        );
        assert_eq!(terminal.num_tokens, Some(expected.num_tokens));
    }
}
