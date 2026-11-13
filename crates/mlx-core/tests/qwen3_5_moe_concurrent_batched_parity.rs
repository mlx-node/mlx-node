//! Real-checkpoint Qwen3.5 MoE continuous-batching parity gate.
//!
//! Text-only requests. Vision-capable checkpoints are valid fixtures. The
//! first gate runs plain autoregressive decode only; the wave gate adds
//! fixed-depth MTP when the checkpoint has MTP heads.
//!
//! `QWEN35_MOE_STAGE2_MODEL_PATH=/abs/qwen3.5-moe cargo test -p mlx-core \
//!   --test qwen3_5_moe_concurrent_batched_parity -- --ignored --nocapture`

use std::path::PathBuf;

use futures::future::join_all;
use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::qwen3_5_moe::persistence::load_with_thread;
use mlx_core::tokenizer::ChatMessage;

/// Both gates flip the process env `MLX_SERVE_FORCE_SERIAL`; they must not
/// run at once.
static ENV_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("QWEN35_MOE_STAGE2_MODEL_PATH")?;
    let path = PathBuf::from(path);
    assert!(path.exists(), "QWEN35_MOE_STAGE2_MODEL_PATH does not exist");
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
    config_with(owner, max_new_tokens, false)
}

fn config_with(owner: &str, max_new_tokens: i32, mtp: bool) -> ChatConfig {
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
        enable_mtp: Some(mtp),
        mtp_adaptive_depth: Some(false),
        include_reasoning: Some(true),
        report_performance: Some(mtp),
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
#[ignore = "needs QWEN35_MOE_STAGE2_MODEL_PATH pointing to a Qwen3.5 MoE checkpoint"]
async fn asymmetric_finish_and_cross_owner_warm_wave_match_serial() {
    let Some(path) = model_path() else {
        eprintln!("skipping: QWEN35_MOE_STAGE2_MODEL_PATH unset");
        return;
    };
    let _env = ENV_GATE.lock().await;

    let model = load_with_thread(&path.to_string_lossy(), None, None)
        .await
        .expect("load Qwen3.5 MoE");
    assert!(
        model.has_block_paged_cache(),
        "gate requires block-paged Qwen3.5 MoE"
    );
    // The switch is read per command, so the load keeps the scheduled plan
    // and only the oracle turns run whole-turn.
    unsafe { std::env::set_var("MLX_SERVE_FORCE_SERIAL", "1") };

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
        "a paged checkpoint advertises the scheduled lane by default"
    );
    let scheduled_solo = model
        .chat_session_start(
            vec![user_message(cases[0].0)],
            Some(config("scheduled-solo", cases[0].1)),
        )
        .await
        .expect("scheduled one-row oracle");
    assert_same(&serial[0], &scheduled_solo, "scheduled one-row oracle");
    model
        .reset_caches()
        .await
        .expect("reset scheduled one-row oracle");
    let scheduled_solo_second = model
        .chat_session_start(
            vec![user_message(cases[1].0)],
            Some(config("scheduled-solo-second", cases[1].1)),
        )
        .await
        .expect("scheduled second one-row oracle");
    assert_same(
        &serial[1],
        &scheduled_solo_second,
        "scheduled second one-row oracle",
    );
    model
        .reset_caches()
        .await
        .expect("reset scheduled second one-row oracle");

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
    let batched = batched
        .into_iter()
        .map(|result| result.expect("asymmetric batched turn"))
        .collect::<Vec<_>>();
    for ((expected, actual), (prompt, _)) in serial.iter().zip(&batched).zip(&cases[..2]) {
        assert_same(expected, actual, prompt);
    }

    // Both completed rows are legal warm residents. A third owner forces a
    // one-row wave while another warm row still exists, catching accidental
    // table-wide recurrent-state selection.
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
        "expected a genuine N=2 sparse-MoE decode, got {}",
        stats.max_batch_occupancy
    );
}

const WAVE_PROMPTS: [(&str, i32); 8] = [
    ("Explain briefly why the sky appears blue.", 24),
    ("Explain briefly why ice floats on water.", 17),
    ("Name three common primary colors.", 21),
    ("Give one sentence about the moon.", 13),
    ("What is two plus three? Answer in words.", 19),
    ("Write a short greeting for a new colleague.", 23),
    ("List two uses of copper.", 15),
    ("Say something kind about rain.", 20),
];

/// MTP cycles and accepted drafts a turn reports.
fn mtp_counts(result: &ChatResult) -> (u32, f64) {
    let performance = result.performance.as_ref().expect("performance");
    let cycles = performance.mtp_cycles.unwrap_or(0);
    let accepted = performance.mtp_mean_accepted_tokens.unwrap_or(0.0) * f64::from(cycles);
    (cycles, accepted.round())
}

/// N = 1, 2, 4 and 8 concurrent text turns at temperature 0. With MTP off
/// every turn gives the serial path's tokens. When the checkpoint has MTP
/// heads, fixed-depth MTP turns give the tokens of the same prompt's
/// scheduled turn run alone: the lanes share a step, never a matmul. The
/// serial MTP path is not that reference: its first cycle follows one AR
/// step and its drafter rebuilds history every cycle, so it verifies other
/// windows, and on a MoE checkpoint a one-ulp change in a router input can
/// pick other experts; its first differing character is printed. Every
/// scheduled MTP turn runs draft cycles, at every width the scheduled wave
/// accepts drafts exactly when the serial oracle does on the same prompts,
/// and the oracle accepts at least one draft.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "needs QWEN35_MOE_STAGE2_MODEL_PATH pointing to a Qwen3.5 MoE checkpoint"]
async fn waves_of_one_two_four_and_eight_match_serial() {
    let Some(path) = model_path() else {
        eprintln!("skipping: QWEN35_MOE_STAGE2_MODEL_PATH unset");
        return;
    };
    let _env = ENV_GATE.lock().await;
    let model = load_with_thread(&path.to_string_lossy(), None, None)
        .await
        .expect("load Qwen3.5 MoE");
    assert!(
        model.has_block_paged_cache(),
        "gate requires block-paged Qwen3.5 MoE"
    );
    let lanes = model.max_concurrent_sequences();
    eprintln!("planned lanes: {lanes}");
    assert!((1..=8).contains(&lanes));
    let modes: &[bool] = if model.has_mtp_weights() {
        &[false, true]
    } else {
        &[false]
    };
    // Every wave runs before the verdict, so one load shows each width.
    let mut differing = Vec::new();
    for &mtp in modes {
        // The switch is read per command, so the load keeps its plan and
        // only the oracle turns run whole-turn.
        unsafe { std::env::set_var("MLX_SERVE_FORCE_SERIAL", "1") };
        let mut serial = Vec::new();
        for (index, (prompt, max_tokens)) in WAVE_PROMPTS.iter().enumerate() {
            serial.push(
                model
                    .chat_session_start(
                        vec![user_message(prompt)],
                        Some(config_with(
                            &format!("serial-{mtp}-{index}"),
                            *max_tokens,
                            mtp,
                        )),
                    )
                    .await
                    .expect("serial oracle"),
            );
            model.reset_caches().await.expect("reset serial oracle");
        }
        unsafe { std::env::remove_var("MLX_SERVE_FORCE_SERIAL") };
        let mut alone = Vec::new();
        if mtp {
            for (index, (prompt, max_tokens)) in WAVE_PROMPTS.iter().enumerate() {
                alone.push(
                    model
                        .chat_session_start(
                            vec![user_message(prompt)],
                            Some(config_with(&format!("alone-{index}"), *max_tokens, mtp)),
                        )
                        .await
                        .expect("scheduled turn alone"),
                );
                model
                    .reset_caches()
                    .await
                    .expect("reset after the lone turn");
            }
        }
        let reference = if mtp { &alone } else { &serial };
        let mut serial_accepted = vec![0.0; serial.len()];
        if mtp {
            for (index, result) in serial.iter().enumerate() {
                serial_accepted[index] = mtp_counts(result).1;
            }
            assert!(
                serial_accepted.iter().sum::<f64>() > 0.0,
                "the serial oracle accepted at least one draft"
            );
        }
        for width in [1usize, 2, 4, 8] {
            let mut scheduled_accepted = 0.0;
            let wave = join_all(WAVE_PROMPTS[..width].iter().enumerate().map(
                |(index, (prompt, max_tokens))| {
                    model.chat_session_start(
                        vec![user_message(prompt)],
                        Some(config_with(
                            &format!("wave-{mtp}-{width}-{index}"),
                            *max_tokens,
                            mtp,
                        )),
                    )
                },
            ))
            .await;
            for (((expected, oracle), actual), (prompt, _)) in reference
                .iter()
                .zip(&serial)
                .zip(wave)
                .zip(&WAVE_PROMPTS[..width])
            {
                let actual = actual.expect("batched turn");
                let label = format!("mtp={mtp} N={width}: {prompt}");
                let first_difference = |reference: &ChatResult| {
                    actual
                        .raw_text
                        .chars()
                        .zip(reference.raw_text.chars())
                        .position(|(left, right)| left != right)
                };
                eprintln!(
                    "{label}: first character differing from the reference {:?}, from the \
                     serial path {:?}",
                    first_difference(expected),
                    first_difference(oracle)
                );
                if actual.text != expected.text
                    || actual.raw_text != expected.raw_text
                    || actual.finish_reason != expected.finish_reason
                    || actual.num_tokens != expected.num_tokens
                {
                    differing.push(label.clone());
                }
                if mtp {
                    let (cycles, accepted) = mtp_counts(&actual);
                    let (serial_cycles, serial_drafts) = mtp_counts(oracle);
                    eprintln!(
                        "{label}: scheduled {cycles} cycles, {accepted} accepted; serial \
                         {serial_cycles} cycles, {serial_drafts} accepted"
                    );
                    assert!(cycles > 0, "no draft cycle for {label}");
                    scheduled_accepted += accepted;
                }
            }
            if mtp {
                let serial_wave: f64 = serial_accepted[..width].iter().sum();
                eprintln!(
                    "mtp=true N={width}: accepted drafts: scheduled {scheduled_accepted}, \
                     serial {serial_wave}"
                );
                assert_eq!(
                    scheduled_accepted > 0.0,
                    serial_wave > 0.0,
                    "N={width}: the scheduled wave accepts drafts exactly when the serial \
                     oracle does"
                );
            }
            model.reset_caches().await.expect("reset after wave");
        }
        let stats = model.scheduler_stats().await.expect("scheduler stats");
        eprintln!(
            "mtp={mtp}: max_batch_occupancy {}",
            stats.max_batch_occupancy
        );
        if !mtp {
            // The occupancy is a running maximum, so only the first pass
            // can prove it.
            assert!(
                stats.max_batch_occupancy >= lanes.min(8),
                "every granted lane decodes in one step"
            );
        }
    }
    assert!(
        differing.is_empty(),
        "turns whose tokens differ from their reference: {differing:?}"
    );
}
