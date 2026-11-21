//! Scheduler-driven Qwen3.5 hybrid-state lifecycle gate.
//!
//! Run with a dense Qwen3.5 checkpoint. Installed vision/MTP weights are
//! allowed because every request explicitly uses text-only plain AR:
//! `QWEN35_STAGE2_MODEL_PATH=/abs/qwen3.5-text cargo test -p mlx-core --test qwen3_5_concurrent_batched_parity -- --ignored --nocapture`

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::future::join_all;
use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::qwen3_5::Qwen3_5Model;
use mlx_core::models::qwen3_5::model::Qwen35LoadOptions;
use mlx_core::tokenizer::ChatMessage;

/// Served context cap for the plain-AR asymmetric parity test (int8 KV, MTP off).
///
/// On 7 GB macOS CI runners (`working_set ≈ 4.67 GiB`, `hard_budget ≈ 3.67 GiB`),
/// `qwen3.5-0.8b-mlx-bf16` holds ~2.30 GiB of active weights (safetensors +
/// finalized caches). Disabling the unused MTP head and capping context to
/// 8,192 brings the 2-lane int8 plan minimum down to ~225 MiB (hundreds of MiB
/// below the ~555 MiB dynamic budget), so `evaluate_qwen35_plan` grants
/// `lanes >= 2` on 7 GB CI runners while still scaling up on larger machines.
const ASYMMETRIC_MAX_POSITION_EMBEDDINGS: u32 = 8_192;

/// Both gates flip the process env `MLX_SERVE_FORCE_SERIAL`; they must not
/// run at once.
static ENV_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("QWEN35_STAGE2_MODEL_PATH")?;
    let path = PathBuf::from(path);
    assert!(path.exists(), "QWEN35_STAGE2_MODEL_PATH does not exist");
    Some(path)
}

/// Optional test-only memory cap (e.g. `MLX_TEST_MAX_MEMORY_BYTES=3936563200`
/// to simulate a 7 GB `macos-26` CI runner's hard budget locally).
fn test_max_memory_bytes() -> Option<i64> {
    let raw = std::env::var("MLX_TEST_MAX_MEMORY_BYTES").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(
        trimmed
            .parse::<i64>()
            .expect("MLX_TEST_MAX_MEMORY_BYTES must be a positive integer byte count"),
    )
}

/// RAII-cleaned checkpoint clone that symlinks weights and writes a modified
/// `config.json` with a capped `max_position_embeddings`, MTP disabled, and no
/// explicit `paged_cache_memory_mb` override.
struct PlainArCheckpointDir(PathBuf);

impl PlainArCheckpointDir {
    fn from_checkpoint(src: &Path, label: &str, max_position_embeddings: u32) -> Self {
        let src = fs::canonicalize(src).expect("canonicalize checkpoint dir");
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dst = std::env::temp_dir().join(format!(
            "mlx-qwen35-parity-{label}-{}-{nanos}",
            std::process::id()
        ));
        if dst.exists() {
            let _ = fs::remove_dir_all(&dst);
        }
        fs::create_dir_all(&dst).expect("create plain-AR checkpoint dir");
        let guard = Self(dst);

        let patch_config = |obj: &mut serde_json::Map<String, serde_json::Value>| {
            obj.insert(
                "max_position_embeddings".to_string(),
                serde_json::Value::from(max_position_embeddings),
            );
            obj.remove("paged_cache_memory_mb");
            obj.insert(
                "mtp_num_hidden_layers".to_string(),
                serde_json::Value::from(0),
            );
            obj.remove("num_nextn_predict_layers");
        };

        for entry in fs::read_dir(&src).expect("read checkpoint dir") {
            let entry = entry.expect("checkpoint dir entry");
            let from = entry.path();
            let to = guard.0.join(entry.file_name());
            if entry.file_name() == "config.json" {
                let raw = fs::read_to_string(&from).expect("read config.json");
                let mut cfg: serde_json::Value =
                    serde_json::from_str(&raw).expect("parse config.json");
                if let Some(obj) = cfg.as_object_mut() {
                    patch_config(obj);
                    if let Some(text_cfg) =
                        obj.get_mut("text_config").and_then(|v| v.as_object_mut())
                    {
                        patch_config(text_cfg);
                    }
                }
                let pretty = serde_json::to_string_pretty(&cfg).expect("serialize config.json");
                fs::write(&to, pretty).expect("write capped config.json");
            } else {
                #[cfg(unix)]
                std::os::unix::fs::symlink(&from, &to).expect("symlink checkpoint asset");
                #[cfg(not(unix))]
                if from.is_file() {
                    fs::copy(&from, &to).expect("copy checkpoint asset");
                }
            }
        }

        guard
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for PlainArCheckpointDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
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
#[ignore = "needs QWEN35_STAGE2_MODEL_PATH pointing to a dense Qwen3.5 checkpoint"]
async fn asymmetric_finish_and_cross_owner_warm_wave_match_serial() {
    let Some(path) = model_path() else {
        eprintln!("skipping: QWEN35_STAGE2_MODEL_PATH unset");
        return;
    };
    let _env = ENV_GATE.lock().await;
    let capped = PlainArCheckpointDir::from_checkpoint(
        &path,
        "asymmetric",
        ASYMMETRIC_MAX_POSITION_EMBEDDINGS,
    );
    let model = Qwen3_5Model::load(
        capped.path().to_string_lossy().into_owned(),
        Some(Qwen35LoadOptions {
            max_memory_bytes: test_max_memory_bytes(),
            ..Default::default()
        }),
    )
    .await
    .expect("load qwen3.5");
    assert!(model.has_block_paged_cache(), "gate requires paged Qwen3.5");
    assert!(
        !model.has_mtp_weights(),
        "asymmetric gate disables MTP in config.json so plain-AR int8 plans >= 2 lanes on 7 GB CI"
    );
    // The switch is read per command, so the load keeps the scheduled plan
    // and only the oracle turns run whole-turn.
    unsafe { std::env::set_var("MLX_SERVE_FORCE_SERIAL", "1") };

    // Use asymmetric decode lengths large enough that even if `batch-0` arrives
    // ahead of `batch-1` and charges `decode_debt_ms` on its solo prefill,
    // `batch-0` still has plenty of decode steps left to overlap with `batch-1`
    // while finishing well before it.
    let cases = [
        ("Explain briefly why the sky appears blue.", 32),
        ("Explain briefly why ice floats on water.", 48),
        ("Name three common primary colors.", 9),
    ];
    let mut serial = Vec::new();
    for (index, (prompt, max_tokens)) in cases.iter().enumerate() {
        let turn = model
            .chat_session_start(
                vec![user_message(prompt)],
                Some(config(&format!("serial-{index}"), *max_tokens)),
            )
            .await
            .expect("serial oracle");
        eprintln!(
            "asymmetric serial[{index}]: num_tokens={} finish_reason={}",
            turn.num_tokens, turn.finish_reason
        );
        serial.push(turn);
        model.reset_caches().await.expect("reset serial oracle");
    }

    unsafe { std::env::remove_var("MLX_SERVE_FORCE_SERIAL") };
    let lanes = model.max_concurrent_sequences();
    eprintln!("asymmetric planned lanes: {lanes}");
    assert!(
        lanes >= 2,
        "a paged checkpoint advertises at least 2 scheduled lanes, got {lanes}"
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
    eprintln!(
        "asymmetric max_batch_occupancy: {}",
        stats.max_batch_occupancy
    );
    assert!(
        stats.max_batch_occupancy >= 2,
        "expected a genuine N=2 hybrid decode, got {}",
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

/// N = 1, 2, 4 and 8 concurrent text turns give the serial path's tokens
/// at temperature 0, with MTP off and, when the checkpoint has MTP heads,
/// with fixed-depth MTP on. The plan's lanes are what the model reports.
/// Every scheduled MTP turn runs draft cycles (an MTP turn that fell back
/// to AR gives the same tokens). The serial path's first cycle follows one
/// AR step and the scheduled path's does not, so their cycles split the
/// same tokens differently and accepted counts need not be equal; at every
/// width the scheduled wave accepts drafts exactly when the serial oracle
/// does on the same prompts, and the oracle accepts at least one draft.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "needs QWEN35_STAGE2_MODEL_PATH pointing to a dense Qwen3.5 checkpoint"]
async fn waves_of_one_two_four_and_eight_match_serial() {
    let Some(path) = model_path() else {
        eprintln!("skipping: QWEN35_STAGE2_MODEL_PATH unset");
        return;
    };
    let _env = ENV_GATE.lock().await;
    let model = Qwen3_5Model::load(
        path.to_string_lossy().into_owned(),
        test_max_memory_bytes().map(|max_memory_bytes| Qwen35LoadOptions {
            max_memory_bytes: Some(max_memory_bytes),
            ..Default::default()
        }),
    )
    .await
    .expect("load qwen3.5");
    assert!(model.has_block_paged_cache(), "gate requires paged Qwen3.5");
    let lanes = model.max_concurrent_sequences();
    eprintln!("planned lanes: {lanes}");
    assert!((1..=8).contains(&lanes));
    let modes: &[bool] = if model.has_mtp_weights() {
        &[false, true]
    } else {
        &[false]
    };
    for &mtp in modes {
        // Process-local ignored gate: the switch is read per command, so the
        // load keeps its plan and only the oracle turns run whole-turn.
        unsafe { std::env::set_var("MLX_SERVE_FORCE_SERIAL", "1") };
        let mut serial = Vec::new();
        for (index, (prompt, max_tokens)) in WAVE_PROMPTS.iter().enumerate() {
            let turn = model
                .chat_session_start(
                    vec![user_message(prompt)],
                    Some(config_with(
                        &format!("serial-{mtp}-{index}"),
                        *max_tokens,
                        mtp,
                    )),
                )
                .await
                .expect("serial oracle");
            if !mtp {
                eprintln!(
                    "wave serial[{index}] (max={max_tokens}): num_tokens={} finish_reason={}",
                    turn.num_tokens, turn.finish_reason
                );
            }
            serial.push(turn);
            model.reset_caches().await.expect("reset serial oracle");
        }
        unsafe { std::env::remove_var("MLX_SERVE_FORCE_SERIAL") };
        let mut serial_accepted = vec![0.0; serial.len()];
        if mtp {
            for (index, result) in serial.iter().enumerate() {
                let (cycles, accepted) = mtp_counts(result);
                eprintln!("serial turn {index}: {cycles} cycles, {accepted} accepted drafts");
                serial_accepted[index] = accepted;
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
            for ((expected, actual), (prompt, _)) in
                serial.iter().zip(wave).zip(&WAVE_PROMPTS[..width])
            {
                let actual = actual.expect("batched turn");
                let label = format!("mtp={mtp} N={width}: {prompt}");
                assert_same(expected, &actual, &label);
                if mtp {
                    let (cycles, accepted) = mtp_counts(&actual);
                    let (serial_cycles, serial_drafts) = mtp_counts(expected);
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
}
