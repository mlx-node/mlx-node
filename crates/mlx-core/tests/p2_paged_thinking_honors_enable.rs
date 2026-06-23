//! P2 gate: the PAGED whole-turn cores must honor `enable_thinking`
//! (resolved from `reasoning_effort` via the single-source ThinkingSetup)
//! instead of the pre-P2 hardcoded `thinking_enabled = true`.
//!
//! Before P2, `paged_turn_sync_core` / `paged_turn_stream_core` hardcoded
//! `let thinking_enabled = true`, so the DEFAULT paged path forced thinking
//! ON regardless of `enable_thinking=false` — the latent bug the roadmap
//! names. After P2 these read `thinking.enabled = resolve(TemplateHonoring)
//! .enabled = resolve_enable_thinking(config).unwrap_or(true)`.
//!
//! `resolve_enable_thinking` (engine/params.rs): reasoning_effort
//! "none"/"low" => Some(false); "medium"/"high" => Some(true); unset => None.
//! So on the PAGED path:
//!   - reasoning_effort "medium"  => thinking ON  => reasoning_tokens > 0
//!   - reasoning_effort "none"    => thinking OFF => reasoning_tokens == 0
//!
//! The "none" assertion is the negative control: it FAILS on P1 (paged
//! hardcoded thinking_enabled=true → reasoning_tokens > 0) and PASSES on P2.
//!
//! qwen3_5 paged is OPT-IN (default OFF), so the test clones the checkpoint
//! dir and forces `use_block_paged_cache: true` (mirrors qwen3_5_session.rs)
//! and asserts `has_block_paged_cache()` before exercising the cores.
//!
//! Gated on `MLX_TEST_MODEL_PATH`. Run with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16 \
//!     PATH=/usr/bin:$PATH SDKROOT=$(xcrun --show-sdk-path) \
//!     cargo test -p mlx-core --test p2_paged_thinking_honors_enable -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_MODEL_PATH` it early-returns and passes trivially so it
//! still compiles as part of `cargo test`.

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
use mlx_core::tokenizer::ChatMessage;

/// Clone the model dir into the workspace target, symlinking weights and
/// copying config.json with `use_block_paged_cache: true` injected.
fn clone_model_dir_paged(src: &Path, suffix: &str) -> Result<PathBuf, String> {
    let pid = std::process::id();
    let workspace_target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set when running cargo test");
            let mut p = PathBuf::from(manifest);
            p.pop();
            p.pop();
            p.join("target")
        });

    let dst = workspace_target.join(format!("p2-paged-thinking-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    let read_dir = fs::read_dir(src).map_err(|e| format!("read_dir({}): {e}", src.display()))?;
    for entry in read_dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_file() {
            if entry.file_name() == "config.json" {
                fs::copy(&from, &to)
                    .map_err(|e| format!("copy({} -> {}): {e}", from.display(), to.display()))?;
            } else {
                std::os::unix::fs::symlink(&from, &to)
                    .map_err(|e| format!("symlink({} -> {}): {e}", from.display(), to.display()))?;
            }
        }
    }

    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path)
        .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
    let mut cfg: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(true);
    cfg["paged_cache_memory_mb"] = serde_json::Value::from(512u32);
    cfg["paged_block_size"] = serde_json::Value::from(16u32);
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty)
        .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;

    Ok(dst)
}

/// Config with a settable `reasoning_effort` + reasoning included so
/// `reasoning_tokens` is populated. `thinking_token_budget` left None so
/// thinking (when ON) is not artificially truncated to zero.
fn chat_config(reasoning_effort: Option<&str>) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(96),
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
        reasoning_effort: reasoning_effort.map(|s| s.to_string()),
        thinking_token_budget: None,
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(true),
        enable_mtp: None,
        mtp_depth: None,
        mtp_adaptive_depth: None,
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
        audio: None,
    }
}

const PROMPT: &str = "Please explain, in a few clear sentences, why the sky \
                      appears blue during the day and turns orange and red near \
                      sunset. Keep the explanation simple and friendly.";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn qwen3_5_paged_honors_enable_thinking() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };
    let src = PathBuf::from(&model_path);
    if !src.exists() {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH does not exist: {}",
            src.display()
        );
        return;
    }

    let paged_dir = match clone_model_dir_paged(&src, "qwen35-dense") {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir with paged forced on: {e}"),
    };

    let model = Qwen3_5Model::load(paged_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load paged Qwen3.5 Dense model");
    assert!(
        model.has_block_paged_cache(),
        "expected the paged adapter to be built after forcing \
         use_block_paged_cache=true, but has_block_paged_cache()==false"
    );

    // --- thinking ON (reasoning_effort=medium → enable_thinking=true) ---
    // Establishes the positive control: the PAGED path DOES think when asked.
    let on = model
        .chat_session_start(
            vec![user_message(PROMPT)],
            Some(chat_config(Some("medium"))),
        )
        .await
        .expect("paged medium-effort chat_session_start failed");
    assert!(
        on.reasoning_tokens > 0,
        "POSITIVE CONTROL FAILED: reasoning_effort=medium on the paged path \
         produced reasoning_tokens={} (expected > 0). If this is 0 the test \
         setup is wrong (model never thinks), making the OFF assertion vacuous. \
         raw_text={:?}",
        on.reasoning_tokens,
        on.raw_text
    );

    // Independent cold session for the second effort.
    tokio::task::block_in_place(|| model.reset_caches()).expect("reset_caches failed");

    // --- thinking OFF (reasoning_effort=none → enable_thinking=false) ---
    // THE P2 GATE: pre-P2 the paged cores hardcoded thinking_enabled=true so
    // this STILL produced reasoning_tokens > 0; post-P2 the paged path honors
    // enable_thinking=false → reasoning_tokens == 0.
    let off = model
        .chat_session_start(vec![user_message(PROMPT)], Some(chat_config(Some("none"))))
        .await
        .expect("paged none-effort chat_session_start failed");
    assert_eq!(
        off.reasoning_tokens, 0,
        "P2 GATE FAILED: reasoning_effort=none on the PAGED path produced \
         reasoning_tokens={} (expected 0). The paged whole-turn core is NOT \
         honoring enable_thinking=false — it is still forcing thinking on \
         (the pre-P2 hardcoded `thinking_enabled = true` bug). raw_text={:?}",
        off.reasoning_tokens, off.raw_text
    );
}
