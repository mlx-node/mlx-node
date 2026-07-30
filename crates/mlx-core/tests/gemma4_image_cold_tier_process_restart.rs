//! Real-weights Gemma4 image recovery across an actual OS-process restart.
//!
//! This is deliberately separate from `gemma4_cold_tier_parity.rs`: that
//! harness loads fresh model instances but they still share one process-global
//! cold manager. Here the ignored parent test launches this test binary twice,
//! first as a writer and then as a reader. The children share only an explicit
//! `MLX_COLD_CACHE_DIR` and a config-patched model clone.
//!
//! Run with a causal Gemma4 E2B checkpoint that advertises image support:
//!
//! ```shell
//! MLX_TEST_GEMMA4_VL_MODEL_PATH=/path/to/gemma4-mlx \
//! MLX_TEST_VLM_IMAGE_PATH=examples/ocr.png \
//! cargo test -p mlx-core --release \
//!   --test gemma4_image_cold_tier_process_restart -- \
//!   --ignored --exact --nocapture --test-threads=1 \
//!   gemma4_image_cold_tier_survives_real_process_restart
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use mlx_core::cold_tier::{cold_cache_drain, cold_cache_stats_snapshot, cold_sidecar_telemetry};
use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::gemma4::model::Gemma4Model;
use mlx_core::tokenizer::ChatMessage;
use napi::bindgen_prelude::Uint8Array;
use serde_json::{Value, json};

const CHILD_ROLE_ENV: &str = "MLX_GEMMA4_IMAGE_RESTART_CHILD_ROLE";
const CHILD_MODEL_DIR_ENV: &str = "MLX_GEMMA4_IMAGE_RESTART_MODEL_DIR";
const CHILD_RESULT_ENV: &str = "MLX_GEMMA4_IMAGE_RESTART_RESULT";
const CAPTURE_TURNS_ENV: &str = "MLX_GEMMA4_IMAGE_RESTART_CAPTURE_TURNS";
const MODEL_ENV: &str = "MLX_TEST_GEMMA4_VL_MODEL_PATH";
const FALLBACK_MODEL_ENV: &str = "MLX_TEST_MODEL_PATH";
const IMAGE_ENV: &str = "MLX_TEST_VLM_IMAGE_PATH";

const PROMPT: &str = "Describe this image in one short sentence, then identify the most \
    prominent visible detail. Keep the answer factual and concise.";

fn target_dir() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.pop();
            path.pop();
            path.join("target")
        })
}

fn unique_dir(label: &str) -> PathBuf {
    target_dir().join(format!(
        "gemma4-image-process-restart-{}-{label}",
        std::process::id()
    ))
}

fn clone_persist_model(src: &Path) -> Result<PathBuf, String> {
    let dst = unique_dir("model");
    if dst.exists() {
        fs::remove_dir_all(&dst)
            .map_err(|error| format!("remove stale clone {}: {error}", dst.display()))?;
    }
    fs::create_dir_all(&dst)
        .map_err(|error| format!("create model clone {}: {error}", dst.display()))?;

    for entry in
        fs::read_dir(src).map_err(|error| format!("read model dir {}: {error}", src.display()))?
    {
        let entry = entry.map_err(|error| format!("read model entry: {error}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_file() {
            if entry.file_name() == "config.json" {
                fs::copy(&from, &to).map_err(|error| {
                    format!("copy {} to {}: {error}", from.display(), to.display())
                })?;
            } else {
                std::os::unix::fs::symlink(&from, &to).map_err(|error| {
                    format!("symlink {} to {}: {error}", from.display(), to.display())
                })?;
            }
        }
    }

    let config_path = dst.join("config.json");
    let mut config: Value = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .map_err(|error| format!("read {}: {error}", config_path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", config_path.display()))?;
    config["use_block_paged_cache"] = Value::Bool(true);
    config["persist_paged_cache"] = Value::Bool(true);
    config["paged_cache_memory_mb"] = Value::from(1024u32);
    config["paged_block_size"] = Value::from(16u32);
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&config)
            .map_err(|error| format!("serialize config: {error}"))?,
    )
    .map_err(|error| format!("write {}: {error}", config_path.display()))?;
    Ok(dst)
}

fn resolve_model() -> Option<PathBuf> {
    let value = std::env::var(MODEL_ENV)
        .or_else(|_| std::env::var(FALLBACK_MODEL_ENV))
        .ok()?;
    let path = PathBuf::from(value);
    assert!(
        path.is_dir(),
        "{MODEL_ENV}/{FALLBACK_MODEL_ENV} must name a checkpoint directory: {}",
        path.display()
    );
    let config_path = path.join("config.json");
    let config: Value = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", config_path.display())),
    )
    .unwrap_or_else(|error| panic!("parse {}: {error}", config_path.display()));
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let overlay = config
        .pointer("/text_config/use_bidirectional_attention")
        .and_then(Value::as_str);
    assert!(
        !model_type.contains("unified") && overlay != Some("vision"),
        "the byte-exact process-restart gate requires a causal Gemma4 E2B checkpoint; \
         unified vision has a documented cache-hit reduction-order drift and needs a \
         tolerance-based logits gate instead (model_type={model_type:?}, overlay={overlay:?})"
    );
    Some(path)
}

fn resolve_image() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os(IMAGE_ENV) {
        let path = PathBuf::from(value);
        assert!(
            path.is_file(),
            "{IMAGE_ENV} must name an image file: {}",
            path.display()
        );
        return Some(path);
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/ocr.png");
    path.is_file().then_some(path)
}

fn image_message(image: &[u8]) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: PROMPT.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: Some(vec![Uint8Array::new(image.to_vec())]),
        audio: None,
    }
}

fn chat_config() -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(32),
        temperature: Some(0.0),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        thinking_token_budget: Some(32),
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(true),
        ..ChatConfig::default()
    }
}

async fn drain_cold_writer() {
    let drained = tokio::task::spawn_blocking(|| cold_cache_drain(20_000))
        .await
        .unwrap_or(false);
    assert!(drained, "cold-cache writer did not drain within 20 seconds");
}

fn result_json(result: &ChatResult) -> Value {
    let stats = cold_cache_stats_snapshot();
    let sidecars = cold_sidecar_telemetry();
    json!({
        "text": result.text,
        "raw_text": result.raw_text,
        "num_tokens": result.num_tokens,
        "prompt_tokens": result.prompt_tokens,
        "cached_tokens": result.cached_tokens,
        "finish_reason": result.finish_reason,
        "cold": {
            "hits": stats.as_ref().map_or(0, |stats| stats.hits),
            "misses": stats.as_ref().map_or(0, |stats| stats.misses),
            "corruptions": stats.as_ref().map_or(0, |stats| stats.corruptions),
            "bytes_written": stats.as_ref().map_or(0, |stats| stats.bytes_written),
            "bytes_restored": stats.as_ref().map_or(0, |stats| stats.bytes_restored),
        },
        "sidecars": {
            "enqueued": sidecars.enqueued,
            "installed": sidecars.installed,
        }
    })
}

fn write_result(result: &ChatResult) {
    let path = PathBuf::from(
        std::env::var(CHILD_RESULT_ENV)
            .unwrap_or_else(|_| panic!("{CHILD_RESULT_ENV} missing in child")),
    );
    fs::write(
        &path,
        serde_json::to_string_pretty(&result_json(result)).expect("serialize child result"),
    )
    .unwrap_or_else(|error| panic!("write child result {}: {error}", path.display()));
}

async fn run_capture_child(model_dir: &Path, image: &[u8]) {
    let model = Gemma4Model::load_from_dir(&model_dir.to_string_lossy(), None)
        .await
        .expect("load Gemma4 capture model");
    assert!(
        model.supports_images(),
        "the configured checkpoint does not advertise Gemma4 image support"
    );

    let max_turns = std::env::var(CAPTURE_TURNS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    assert!(max_turns > 0, "{CAPTURE_TURNS_ENV} must be positive");
    let trace_path = PathBuf::from(
        std::env::var("MLX_INFERENCE_TRACE_FILE")
            .expect("capture child needs MLX_INFERENCE_TRACE_FILE"),
    );
    let mut baseline: Option<ChatResult> = None;
    for turn in 1..=max_turns {
        let result = model
            .chat_session_start(vec![image_message(image)], Some(chat_config()))
            .await
            .unwrap_or_else(|error| panic!("capture image turn {turn} failed: {error}"));
        assert!(
            result.num_tokens > 0,
            "capture image turn {turn} emitted no tokens"
        );
        if let Some(first) = baseline.as_ref() {
            assert_eq!(
                result.text, first.text,
                "capture warm turn {turn} changed deterministic output"
            );
            assert_eq!(
                result.num_tokens, first.num_tokens,
                "capture warm turn {turn} changed token count"
            );
        } else {
            baseline = Some(result.clone());
        }
        drain_cold_writer().await;

        let captured = fs::read_to_string(&trace_path)
            .unwrap_or_default()
            .contains("sliding_cold_sidecar_capture_enqueued media=image");
        eprintln!(
            "Gemma4 image cold capture turn {turn}/{max_turns}: cached={} sidecar_captured={captured}",
            result.cached_tokens
        );
        if captured {
            break;
        }
    }
    write_result(&baseline.expect("at least one capture turn"));
}

async fn run_restore_child(model_dir: &Path, image: &[u8]) {
    let model = Gemma4Model::load_from_dir(&model_dir.to_string_lossy(), None)
        .await
        .expect("load Gemma4 restore model");
    let result = model
        .chat_session_start(vec![image_message(image)], Some(chat_config()))
        .await
        .expect("restore image turn failed");
    drain_cold_writer().await;
    write_result(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "helper child; launched by gemma4_image_cold_tier_survives_real_process_restart"]
async fn gemma4_image_restart_child() {
    let Ok(role) = std::env::var(CHILD_ROLE_ENV) else {
        eprintln!("skipping helper: {CHILD_ROLE_ENV} unset");
        return;
    };
    let model_dir = PathBuf::from(
        std::env::var(CHILD_MODEL_DIR_ENV)
            .unwrap_or_else(|_| panic!("{CHILD_MODEL_DIR_ENV} missing in child")),
    );
    let image_path = resolve_image().expect("child could not resolve test image");
    let image = fs::read(&image_path)
        .unwrap_or_else(|error| panic!("read image {}: {error}", image_path.display()));

    match role.as_str() {
        "capture" => run_capture_child(&model_dir, &image).await,
        "restore" => run_restore_child(&model_dir, &image).await,
        other => panic!("unknown {CHILD_ROLE_ENV}={other:?}"),
    }
}

fn run_child(
    role: &str,
    model_dir: &Path,
    image_path: &Path,
    cold_root: &Path,
    trace_path: &Path,
    result_path: &Path,
) -> Output {
    Command::new(std::env::current_exe().expect("current integration-test binary"))
        .env(CHILD_ROLE_ENV, role)
        .env(CHILD_MODEL_DIR_ENV, model_dir)
        .env(CHILD_RESULT_ENV, result_path)
        .env(IMAGE_ENV, image_path)
        .env("MLX_COLD_CACHE_DIR", cold_root)
        .env("MLX_INFERENCE_TRACE", "1")
        .env("MLX_INFERENCE_TRACE_FILE", trace_path)
        .arg("--ignored")
        .arg("--exact")
        .arg("gemma4_image_restart_child")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .output()
        .unwrap_or_else(|error| panic!("launch {role} child: {error}"))
}

fn assert_child_success(role: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{role} child failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(
        &fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read result {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("parse result {}: {error}", path.display()))
}

fn trace_field(line: &str, field: &str) -> Option<u32> {
    line.split_whitespace()
        .find_map(|token| token.strip_prefix(field)?.parse().ok())
}

fn value_u64(value: &Value, pointer: &str) -> u64 {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("missing numeric result field {pointer}: {value}"))
}

fn assert_capture_trace(trace: &str) {
    let captures: Vec<&str> = trace
        .lines()
        .filter(|line| {
            line.contains("sliding_cold_sidecar_capture_enqueued") && line.contains("media=image")
        })
        .collect();
    assert!(
        !captures.is_empty(),
        "writer never persisted an image sliding sidecar; capture trace:\n{trace}"
    );
    assert!(
        captures.iter().any(|line| {
            let boundary = trace_field(line, "boundary_tokens=");
            let image_end = trace_field(line, "last_image_exclusive=");
            matches!((boundary, image_end), (Some(boundary), Some(image_end)) if image_end > 0 && boundary >= image_end)
        }),
        "no persisted sliding sidecar boundary safely covered the image span:\n{}",
        captures.join("\n")
    );
}

fn assert_restore_trace(trace: &str, result: &Value) {
    let cached_result = value_u64(result, "/cached_tokens") as u32;
    let prepared: Vec<&str> = trace
        .lines()
        .filter(|line| {
            line.contains("sliding_prefix_prepare_done") && line.contains("state=cold_sidecar")
        })
        .collect();
    assert_eq!(
        prepared.len(),
        1,
        "reader must install exactly one cold sliding sidecar:\n{}",
        prepared.join("\n")
    );
    let cached = trace_field(prepared[0], "cached_prefix_tokens=")
        .expect("sidecar trace missing cached_prefix_tokens");
    let primed = trace_field(prepared[0], "primed_prefix_tokens=")
        .expect("sidecar trace missing primed_prefix_tokens");
    let replay = trace_field(prepared[0], "replay_delta_tokens=")
        .expect("sidecar trace missing replay_delta_tokens");
    assert_eq!(cached, cached_result);
    assert_eq!(primed, cached, "restored sidecar must back the full prefix");
    assert_eq!(replay, 0, "reader replayed cached image prefix tokens");

    let skips: Vec<&str> = trace
        .lines()
        .filter(|line| line.contains("gemma4 vlm_vision_tower_skip"))
        .collect();
    assert_eq!(
        skips.len(),
        1,
        "reader must skip the vision tower exactly once:\n{}",
        skips.join("\n")
    );
    let skip_cached = trace_field(skips[0], "cached_prefix_tokens=")
        .expect("vision skip trace missing cached_prefix_tokens");
    let image_end = trace_field(skips[0], "last_image_exclusive=")
        .expect("vision skip trace missing last_image_exclusive");
    assert_eq!(skip_cached, cached_result);
    assert!(
        image_end > 0 && skip_cached >= image_end,
        "cached prefix {skip_cached} does not cover image end {image_end}"
    );
}

#[test]
#[ignore = "needs a real Gemma4 image checkpoint; spawns capture and restore OS processes"]
fn gemma4_image_cold_tier_survives_real_process_restart() {
    let Some(source_model) = resolve_model() else {
        eprintln!("skipping: set {MODEL_ENV} (or {FALLBACK_MODEL_ENV})");
        return;
    };
    let Some(image_path) = resolve_image() else {
        eprintln!("skipping: set {IMAGE_ENV} or add examples/ocr.png");
        return;
    };

    let model_dir = clone_persist_model(&source_model).expect("clone persist model");
    let cold_root = unique_dir("cold-root");
    let capture_trace = unique_dir("capture.trace");
    let restore_trace = unique_dir("restore.trace");
    let capture_result_path = unique_dir("capture.json");
    let restore_result_path = unique_dir("restore.json");
    fs::create_dir_all(&cold_root).expect("create cold root");

    let capture_output = run_child(
        "capture",
        &model_dir,
        &image_path,
        &cold_root,
        &capture_trace,
        &capture_result_path,
    );
    assert_child_success("capture", &capture_output);
    let capture_result = read_json(&capture_result_path);
    let capture_trace_text = fs::read_to_string(&capture_trace).expect("read capture trace");
    assert_capture_trace(&capture_trace_text);
    assert!(
        value_u64(&capture_result, "/cold/bytes_written") > 0,
        "capture child wrote no cold-cache bytes: {capture_result}"
    );
    assert!(
        value_u64(&capture_result, "/sidecars/enqueued") > 0,
        "capture child enqueued no sidecar: {capture_result}"
    );

    // The first child is fully gone before this command is launched. The
    // reader cannot inherit its OnceLock, manager index, model, or hot cache.
    let restore_output = run_child(
        "restore",
        &model_dir,
        &image_path,
        &cold_root,
        &restore_trace,
        &restore_result_path,
    );
    assert_child_success("restore", &restore_output);
    let restore_result = read_json(&restore_result_path);
    let restore_trace_text = fs::read_to_string(&restore_trace).expect("read restore trace");
    assert_restore_trace(&restore_trace_text, &restore_result);

    assert!(
        value_u64(&restore_result, "/cached_tokens") > 0,
        "reader restored no prefix: {restore_result}"
    );
    assert!(
        value_u64(&restore_result, "/cold/hits") > 0,
        "reader recorded no cold-tier hit: {restore_result}"
    );
    assert_eq!(
        value_u64(&restore_result, "/cold/corruptions"),
        0,
        "reader fell open over a corrupt object: {restore_result}"
    );
    assert!(
        value_u64(&restore_result, "/sidecars/installed") > 0,
        "reader decoded but did not install the sliding sidecar: {restore_result}"
    );
    assert_eq!(
        capture_result["text"], restore_result["text"],
        "greedy output changed across the process restart"
    );
    assert_eq!(
        capture_result["num_tokens"], restore_result["num_tokens"],
        "generated token count changed across the process restart"
    );

    eprintln!(
        "Gemma4 image process-restart PASS: cached={} hits={} sidecar installed, \
         vision tower skipped, output matched",
        value_u64(&restore_result, "/cached_tokens"),
        value_u64(&restore_result, "/cold/hits")
    );

    for path in [
        &model_dir,
        &cold_root,
        &capture_trace,
        &restore_trace,
        &capture_result_path,
        &restore_result_path,
    ] {
        if path.is_dir() {
            let _ = fs::remove_dir_all(path);
        } else {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod assertion_tests {
    use super::*;

    fn result(cached: u32) -> Value {
        json!({"cached_tokens": cached})
    }

    #[test]
    fn process_gate_trace_assertions_accept_full_image_recovery() {
        assert_capture_trace(
            "[MLX_TRACE] gemma4 sliding_cold_sidecar_capture_enqueued media=image \
             boundary_tokens=336 last_image_exclusive=304",
        );
        assert_restore_trace(
            "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=cold_sidecar \
             cached_prefix_tokens=336 primed_prefix_tokens=336 replay_delta_tokens=0\n\
             [MLX_TRACE] gemma4 vlm_vision_tower_skip cached_prefix_tokens=336 \
             last_image_exclusive=304 suffix_tokens=14",
            &result(336),
        );
    }

    #[test]
    #[should_panic(expected = "does not cover image end")]
    fn process_gate_rejects_vision_skip_before_image_end() {
        assert_restore_trace(
            "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=cold_sidecar \
             cached_prefix_tokens=288 primed_prefix_tokens=288 replay_delta_tokens=0\n\
             [MLX_TRACE] gemma4 vlm_vision_tower_skip cached_prefix_tokens=288 \
             last_image_exclusive=304 suffix_tokens=62",
            &result(288),
        );
    }

    #[test]
    #[should_panic(expected = "restored sidecar must back the full prefix")]
    fn process_gate_rejects_partial_sidecar_install() {
        assert_restore_trace(
            "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=cold_sidecar \
             cached_prefix_tokens=336 primed_prefix_tokens=320 replay_delta_tokens=16\n\
             [MLX_TRACE] gemma4 vlm_vision_tower_skip cached_prefix_tokens=336 \
             last_image_exclusive=304 suffix_tokens=14",
            &result(336),
        );
    }
}
