//! Real-NemotronH continuous-batching token parity and occupancy gate.
//!
//! The serial oracle runs through the scheduler one request at a time, so its
//! decode carries the same incremental mamba state the batched lane stacks into
//! [N, ...] rows. MTP is explicitly disabled so the ordered-barrier path cannot
//! mask the scheduled lane.
//!
//! Reads `MLX_TEST_NEMOTRON_H_MODEL_PATH`; the sibling
//! `nemotron_h_paged_vs_flat_parity.rs` reads the generic `MLX_TEST_MODEL_PATH`
//! instead, so set both or one silently skips.

use std::path::PathBuf;

use futures::future::join_all;
use mlx_core::engine::SchedulerStatsJs;
use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::nemotron_h::model::NemotronHModel;
use mlx_core::tokenizer::ChatMessage;

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("MLX_TEST_NEMOTRON_H_MODEL_PATH")?;
    let path = PathBuf::from(path);
    assert!(
        path.exists(),
        "MLX_TEST_NEMOTRON_H_MODEL_PATH does not exist"
    );
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
        enable_mtp: Some(false),
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

/// Minimum number of decode steps that must have carried >= 2 live rows. A
/// peak-occupancy check would pass on ONE lucky overlapping step; a third of
/// the window is the smallest threshold incidental overlap cannot reach.
const MIN_BATCHED_DECODE_STEPS: f64 = 8.0;

/// Steps recorded at occupancy >= 2 in the cumulative decode histogram.
fn batched_decode_steps(stats: &SchedulerStatsJs) -> f64 {
    stats
        .decode_batch_occupancy_hist
        .iter()
        .filter(|bucket| bucket.occupancy >= 2)
        .map(|bucket| bucket.steps)
        .sum()
}

fn occupancy_report(stats: &SchedulerStatsJs) -> Vec<(u32, f64)> {
    stats
        .decode_batch_occupancy_hist
        .iter()
        .map(|bucket| (bucket.occupancy, bucket.steps))
        .collect()
}

async fn load_model_or_skip() -> Option<NemotronHModel> {
    let path = model_path()?;
    let model = NemotronHModel::load(path.to_string_lossy().into_owned())
        .await
        .expect("load nemotron_h");
    assert!(
        model.has_block_paged_cache(),
        "gate requires paged NemotronH"
    );
    assert!(
        model.max_concurrent_sequences() >= 2,
        "NemotronH scheduler must advertise a batched lane"
    );
    Some(model)
}

/// Build the serial oracle, then require the concurrent run to be
/// token-identical and to have a genuinely batched decode window.
async fn serial_then_batched_parity(
    model: &NemotronHModel,
    prompts: &[String],
    tag: &str,
) -> Vec<ChatResult> {
    let mut serial = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let result = model
            .chat_session_start(
                vec![user_message(prompt)],
                Some(config(&format!("{tag}-serial-{index}"))),
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
            Some(config(&format!("{tag}-batch-{index}"))),
        )
    }))
    .await;
    for ((expected, actual), prompt) in serial.iter().zip(batched).zip(prompts) {
        assert_same(expected, &actual.expect("batched NemotronH turn"), prompt);
    }

    let stats = model.scheduler_stats().await.expect("scheduler stats");
    let batched_steps = batched_decode_steps(&stats);
    assert!(
        stats.max_batch_occupancy >= 2,
        "{tag}: expected a real NemotronH N=2 decode step, max={} hist={:?}",
        stats.max_batch_occupancy,
        occupancy_report(&stats)
    );
    assert!(
        batched_steps >= MIN_BATCHED_DECODE_STEPS,
        "{tag}: only {batched_steps} decode steps carried >= 2 rows (need \
         {MIN_BATCHED_DECODE_STEPS}); the requests were effectively \
         serialized, so the batched-vs-serial parity above compared the \
         scheduler against itself. hist={:?}",
        occupancy_report(&stats)
    );
    assert!(stats.block_capacity > 0, "live block telemetry is required");
    assert_eq!(stats.rows_alloc_evicted, 0.0);
    println!(
        "{tag}: {batched_steps} batched decode steps, hist={:?}",
        occupancy_report(&stats)
    );
    serial
}

/// Deterministic long prompt sized so a PAIR cannot fit the scheduler's
/// per-step token budget. Lines are all distinct (no n-gram cutoff
/// interaction) and the two topics differ, so the rows are not copies.
fn long_prompt(topic: &str, lines: usize) -> String {
    let mut s = String::with_capacity(lines * 96);
    s.push_str("Read the following instrument log, then answer the question at the end.\n\n");
    for i in 0..lines {
        s.push_str(&format!(
            "line {i:04}: sensor {topic} reported value {} at offset {} with status nominal\n",
            (i * 37) % 991,
            i * 13
        ));
    }
    s.push_str("\nIn one short sentence: what status did every line report?");
    s
}

/// Short-prompt case: single-step prefill per row, long shared decode window.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_NEMOTRON_H_MODEL_PATH pointing to a real NemotronH checkpoint"]
async fn carried_state_serial_and_n2_batch_are_token_identical() {
    let Some(model) = load_model_or_skip().await else {
        eprintln!("skipping: MLX_TEST_NEMOTRON_H_MODEL_PATH unset");
        return;
    };

    let prompts = [
        "Give a concise explanation of why leaves look green.".to_string(),
        "Give a concise explanation of why ocean tides change.".to_string(),
    ];
    serial_then_batched_parity(&model, &prompts, "short").await;
}

/// Long-prompt case: the pair exceeds the scheduler's per-step token budget, so
/// admission must split the prefill and the chunk-aligned multi-row path runs.
/// The short case never reaches that budget, so it cannot see a split that
/// mis-slices a Mamba-2 chunk across a step boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_NEMOTRON_H_MODEL_PATH pointing to a real NemotronH checkpoint"]
async fn long_prompt_multi_row_prefill_serial_and_n2_batch_are_token_identical() {
    let Some(model) = load_model_or_skip().await else {
        eprintln!("skipping: MLX_TEST_NEMOTRON_H_MODEL_PATH unset");
        return;
    };

    let prompts = [long_prompt("alpha", 120), long_prompt("bravo", 120)];
    let serial = serial_then_batched_parity(&model, &prompts, "long").await;

    // ANTI-VACUITY: if the prompts render short, no prefill split happens and
    // this is just a slower copy of the short case.
    let total: u32 = serial.iter().map(|r| r.prompt_tokens).sum();
    for (index, result) in serial.iter().enumerate() {
        assert!(
            result.prompt_tokens > 1024,
            "long prompt #{index} rendered only {} tokens; a single row must \
             already be substantial for the pair to exceed the budget",
            result.prompt_tokens
        );
    }
    assert!(
        total > 2048,
        "the long-prompt PAIR rendered {total} tokens, which fits the \
         scheduler's 2048-token per-step budget in one step — the multi-row \
         prefill split was never exercised; raise the line count"
    );
    println!(
        "long case: prompt tokens per row = {:?}",
        serial.iter().map(|r| r.prompt_tokens).collect::<Vec<_>>()
    );
}
