//! Scheduler-driven Qwen3.5 hybrid-state lifecycle gate.
//!
//! Run with a dense Qwen3.5 checkpoint. Installed vision/MTP weights are
//! allowed because every request explicitly uses text-only plain AR:
//! `QWEN35_STAGE2_MODEL_PATH=/abs/qwen3.5-text cargo test -p mlx-core --test qwen3_5_concurrent_batched_parity -- --ignored --nocapture`

use std::path::PathBuf;

use futures::future::join_all;
use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::qwen3_5::persistence::load_with_thread;
use mlx_core::tokenizer::ChatMessage;

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("QWEN35_STAGE2_MODEL_PATH")?;
    let path = PathBuf::from(path);
    assert!(path.exists(), "QWEN35_STAGE2_MODEL_PATH does not exist");
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

fn config(owner: &str, max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        cache_salt: None,
        cache_owner_id: Some(owner.to_string()),
        cache_root_owner_id: Some(owner.to_string()),
        max_new_tokens: Some(max_new_tokens),
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

fn assert_same(expected: &ChatResult, actual: &ChatResult, label: &str) {
    assert_eq!(actual.text, expected.text, "text mismatch for {label}");
    assert_eq!(
        actual.raw_text, expected.raw_text,
        "raw_text mismatch for {label}"
    );
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert_eq!(actual.num_tokens, expected.num_tokens);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs QWEN35_STAGE2_MODEL_PATH pointing to a dense Qwen3.5 checkpoint"]
async fn asymmetric_finish_and_cross_owner_warm_wave_match_serial() {
    let Some(path) = model_path() else {
        eprintln!("skipping: QWEN35_STAGE2_MODEL_PATH unset");
        return;
    };
    // Process-local ignored gate: set before the model thread and OnceLock are
    // created. No sibling test runs in this binary.
    unsafe { std::env::set_var("MLX_CONTINUOUS_BATCHING", "1") };
    unsafe { std::env::set_var("MLX_SERVE_FORCE_SERIAL", "1") };
    let model = load_with_thread(&path.to_string_lossy())
        .await
        .expect("load qwen3.5");
    assert!(model.has_block_paged_cache(), "gate requires paged Qwen3.5");

    let cases = [
        ("Explain briefly why the sky appears blue.", 7),
        ("Explain briefly why ice floats on water.", 15),
        ("Name three common primary colors.", 9),
    ];
    let mut serial = Vec::new();
    for (index, (prompt, max_tokens)) in cases.iter().enumerate() {
        serial.push(
            model
                .chat_session_start(
                    vec![user_message(prompt)],
                    Some(config(&format!("serial-{index}"), *max_tokens)),
                )
                .await
                .expect("serial oracle"),
        );
        model.reset_caches().await.expect("reset serial oracle");
    }

    unsafe { std::env::remove_var("MLX_SERVE_FORCE_SERIAL") };
    assert!(
        model.max_concurrent_sequences() >= 2,
        "opted-in checkpoint must advertise the scheduled lane"
    );
    let batched = join_all(
        cases[..2]
            .iter()
            .enumerate()
            .map(|(index, (prompt, max_tokens))| {
                model.chat_session_start(
                    vec![user_message(prompt)],
                    Some(config(&format!("batch-{index}"), *max_tokens)),
                )
            }),
    )
    .await;
    for ((expected, actual), (prompt, _)) in serial.iter().zip(batched).zip(&cases[..2]) {
        assert_same(expected, &actual.expect("asymmetric batched turn"), prompt);
    }

    // Both completed rows are now legal warm residents. A third owner creates
    // a sequential one-row wave while one old warm row remains in the table.
    // The decode must select its own row instead of demanding table_len == 1.
    let third = model
        .chat_session_start(
            vec![user_message(cases[2].0)],
            Some(config("cross-owner-third", cases[2].1)),
        )
        .await
        .expect("cross-owner warm wave");
    assert_same(&serial[2], &third, cases[2].0);

    let stats = model.scheduler_stats().await.expect("scheduler stats");
    assert!(
        stats.max_batch_occupancy >= 2,
        "expected a genuine N=2 hybrid decode, got {}",
        stats.max_batch_occupancy
    );
}
