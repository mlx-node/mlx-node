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

struct RaggedStepGuard;

impl RaggedStepGuard {
    fn enable() -> Self {
        // The uniform model thread is dropped before this process-level test
        // switch. The replacement resident model reads the mode once during
        // construction, giving us a same-binary three-way oracle.
        unsafe { std::env::set_var("MLX_SCHED_RAGGED_STEP", "1") };
        Self
    }
}

impl Drop for RaggedStepGuard {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("MLX_SCHED_RAGGED_STEP") };
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

fn config(owner: &str, index: usize) -> ChatConfig {
    // Deliberately vary penalty state by row. No-op, identical penalties
    // cannot detect a batched implementation that accidentally applies one
    // request's token history/configuration to another row.
    let penalized = index.is_multiple_of(2);
    ChatConfig {
        cache_salt: None,
        cache_owner_id: Some(owner.to_string()),
        cache_root_owner_id: Some(owner.to_string()),
        max_new_tokens: Some(if penalized { 12 } else { 16 }),
        temperature: Some(0.0),
        repetition_penalty: Some(if penalized { 1.15 } else { 1.0 }),
        presence_penalty: Some(if penalized { 0.2 } else { 0.0 }),
        frequency_penalty: Some(if penalized { 0.1 } else { 0.0 }),
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
    receiver: tokio::sync::mpsc::Receiver<napi::Result<ChatStreamChunk>>,
) -> ChatStreamChunk {
    drain_stream_outcome(receiver)
        .await
        .expect("streaming scheduler row failed")
}

async fn drain_stream_outcome(
    mut receiver: tokio::sync::mpsc::Receiver<napi::Result<ChatStreamChunk>>,
) -> Result<ChatStreamChunk, String> {
    while let Some(chunk) = receiver.recv().await {
        let chunk = chunk.map_err(|error| error.reason)?;
        if chunk.done {
            return Ok(chunk);
        }
    }
    Err("stream closed without a terminal chunk".to_string())
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
                    Some(config(&format!("serial-{index}"), index)),
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
            Some(config(&format!("batch-{index}"), index)),
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

    model
        .reset_caches()
        .await
        .expect("reset before uniform streams");
    let mut uniform_receivers = Vec::new();
    for (index, prompt) in prompts[..2].iter().enumerate() {
        let (_handle, receiver) = model
            .chat_stream_session_start_for_test(
                vec![user_message(prompt)],
                Some(config(&format!("uniform-stream-{index}"), index)),
            )
            .expect("dispatch uniform stream");
        uniform_receivers.push(receiver);
    }
    let uniform_terminals = join_all(uniform_receivers.into_iter().map(drain_stream)).await;
    for ((expected, terminal), prompt) in serial.iter().zip(uniform_terminals).zip(prompts) {
        assert_eq!(
            terminal.raw_text.as_deref(),
            Some(expected.raw_text.as_str()),
            "uniform stream raw_text mismatch for {prompt:?}"
        );
        assert_eq!(
            terminal.finish_reason.as_deref(),
            Some(expected.finish_reason.as_str())
        );
        assert_eq!(terminal.num_tokens, Some(expected.num_tokens));
    }

    // Cancellation is request-local in a real scheduled wave. Keep one row in
    // a long prefill, cancel it after both commands have been dispatched, and
    // require the healthy twin to remain byte-identical to its serial oracle.
    model
        .reset_caches()
        .await
        .expect("reset before cancel twin");
    let cancelled_prompt = format!(
        "Read these notes, then answer briefly: {}",
        "scheduler cache tensor kernel ".repeat(1_024)
    );
    let (cancel_handle, cancel_receiver) = model
        .chat_stream_session_start_for_test(
            vec![user_message(&cancelled_prompt)],
            Some(config("cancelled-twin", 0)),
        )
        .expect("dispatch cancellable twin");
    let (_healthy_handle, healthy_receiver) = model
        .chat_stream_session_start_for_test(
            vec![user_message(prompts[1])],
            Some(config("healthy-twin", 1)),
        )
        .expect("dispatch healthy twin");
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    cancel_handle.cancel();
    let (cancelled_outcome, healthy_outcome) = tokio::join!(
        drain_stream_outcome(cancel_receiver),
        drain_stream_outcome(healthy_receiver)
    );
    assert!(
        cancelled_outcome.is_err()
            || cancelled_outcome
                .as_ref()
                .is_ok_and(|chunk| chunk.finish_reason.as_deref() == Some("cancelled")),
        "cancelled row must fail or terminate as cancelled: {cancelled_outcome:?}"
    );
    let healthy = healthy_outcome.expect("healthy twin must not inherit peer cancellation");
    assert_eq!(
        healthy.raw_text.as_deref(),
        Some(serial[1].raw_text.as_str())
    );
    assert_eq!(
        healthy.finish_reason.as_deref(),
        Some(serial[1].finish_reason.as_str())
    );
    assert_eq!(healthy.num_tokens, Some(serial[1].num_tokens));

    drop(model);
    let ragged_guard = RaggedStepGuard::enable();
    let ragged_model = load_with_thread(&path.to_string_lossy())
        .await
        .expect("load ragged qwen3");
    let ragged_started = Instant::now();
    let ragged = join_all(prompts.iter().enumerate().map(|(index, prompt)| {
        ragged_model.chat_session_start(
            vec![user_message(prompt)],
            Some(config(&format!("ragged-{index}"), index)),
        )
    }))
    .await;
    let ragged_elapsed = ragged_started.elapsed();
    for ((expected, actual), prompt) in serial.iter().zip(ragged).zip(prompts) {
        assert_same(expected, &actual.expect("ragged turn"), prompt);
    }
    let ragged_stats = ragged_model
        .scheduler_stats()
        .await
        .expect("ragged scheduler stats");
    assert!(
        ragged_stats.max_batch_occupancy >= 8,
        "expected a real N=8 ragged decode step, stats max={} hist={:?}",
        ragged_stats.max_batch_occupancy,
        ragged_stats
            .decode_batch_occupancy_hist
            .iter()
            .map(|bucket| (bucket.occupancy, bucket.steps))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "stage2 ragged N=8 wall={:.2}ms, uniform/ragged={:.2}x",
        ragged_elapsed.as_secs_f64() * 1_000.0,
        batched_elapsed.as_secs_f64() / ragged_elapsed.as_secs_f64()
    );

    ragged_model
        .reset_caches()
        .await
        .expect("reset before streams");
    let mut receivers = Vec::new();
    for (index, prompt) in prompts[..2].iter().enumerate() {
        let (_handle, receiver) = ragged_model
            .chat_stream_session_start_for_test(
                vec![user_message(prompt)],
                Some(config(&format!("stream-{index}"), index)),
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
    drop(ragged_model);
    drop(ragged_guard);
}
