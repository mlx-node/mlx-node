//! Gated session-reset regression test for Qwen3.5 Dense on the PAGED
//! adapter.
//!
//! Proves that an explicit `reset_caches` (`ResetScope::Command`) purges
//! the paged adapter's content-addressed prefix cache, so a same-prompt
//! rerun replays a COLD full-prompt prefill (`cached_tokens == 0`)
//! instead of the prefix-hit 1-token-suffix prefill — whose different
//! bf16 reduction order can flip a greedy near-tie. (codex follow-up to
//! the lfm2 fix in 47d8dc53; qwen3_5 shares the identical paged adapter
//! lifecycle.)
//!
//! qwen3_5 paged is OPT-IN (default OFF), so the test clones the
//! checkpoint dir and forces `use_block_paged_cache: true` in
//! config.json (only config.json is copied; weights are symlinked), then
//! asserts `has_block_paged_cache()` to confirm the adapter actually
//! built before exercising the reset.
//!
//! Gated on `MLX_TEST_MODEL_PATH`. Run with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16 \
//!     PATH=/usr/bin:$PATH SDKROOT=$(xcrun --show-sdk-path) \
//!     cargo test -p mlx-core --test qwen3_5_session -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_MODEL_PATH` the test early-returns and passes
//! trivially so it still compiles as part of `cargo test`.

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
use mlx_core::tokenizer::ChatMessage;

/// Clone the model dir into the workspace target, symlinking the weight
/// files and copying config.json with `use_block_paged_cache: true`
/// injected. Mirrors `qwen3_5_paged_vs_flat_parity.rs::clone_model_dir`.
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

    let dst = workspace_target.join(format!("reset-purge-{pid}-{suffix}"));
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
            let name = entry.file_name();
            if name == "config.json" {
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

fn chat_config_default(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new_tokens),
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
        reasoning_effort: None,
        thinking_token_budget: Some(32),
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

/// Explicit `reset_caches` (`ResetScope::Command`) must purge the paged
/// adapter's content-addressed prefix cache, so a same-prompt rerun
/// cold-prefills (`cached_tokens == 0`) rather than taking the prefix-hit
/// suffix-prefill path (which would report `cached_tokens > 0` and can
/// flip a greedy near-tie via a different bf16 reduction order).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn qwen3_5_session_reset_purges_prefix_cache_cold_prefill() {
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

    // Force the paged adapter ON (qwen3_5 is opt-in / default OFF).
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

    // The prompt MUST render to MORE than one paged block
    // (paged_block_size=16 above) for this assertion to bite: the prefix
    // lookup is capped at `max_cache_hit_tokens = total_budget - 1 =
    // prompt_len - 1` (model.rs:2812) and `find_longest_cache_hit` only
    // matches COMPLETE blocks, so a single-block (<=16-token) prompt
    // always reports `cached_tokens == 0` on turn 2 regardless of the
    // purge. This multi-sentence prompt is comfortably >= 33 tokens
    // (>= 2 full blocks), so WITHOUT the purge turn 2 takes a >= 16-token
    // prefix hit (`cached_tokens > 0`) and WITH the purge cold-prefills
    // (`cached_tokens == 0`).
    let prompt = "Please explain, in a few clear sentences, why the sky \
                  appears blue during the day and turns orange and red near \
                  sunset. Keep the explanation simple and friendly.";

    // Turn 1: cold session start. Primes the prefix cache.
    let r1 = model
        .chat_session_start(vec![user_message(prompt)], Some(chat_config_default(32)))
        .await
        .expect("turn 1 chat_session_start failed");
    assert_eq!(
        r1.cached_tokens, 0,
        "turn 1 should cold-start: cached_tokens={}",
        r1.cached_tokens
    );

    // Explicit Command reset via the sync NAPI method. block_in_place:
    // reset_caches blocks on blocking_recv, which panics on a tokio
    // worker thread (see lfm2_session.rs + 8d5283a7 precedent).
    tokio::task::block_in_place(|| model.reset_caches()).expect("reset_caches failed");

    // Turn 2: rerun the IDENTICAL prompt as a fresh session start.
    let r2 = model
        .chat_session_start(vec![user_message(prompt)], Some(chat_config_default(32)))
        .await
        .expect("turn 2 chat_session_start after reset_caches failed");

    // PRIMARY deterministic gate (independent of any bf16 near-tie):
    // without the purge turn-1's full blocks survive content-addressed →
    // cached_tokens > 0; with the purge they are drained → cold prefill.
    assert_eq!(
        r2.cached_tokens, 0,
        "post-Command-reset same-prompt turn must cold-prefill (prefix cache \
         purged), but cached_tokens={} (turn1 prompt_tokens={}, turn2 \
         prompt_tokens={})",
        r2.cached_tokens, r1.prompt_tokens, r2.prompt_tokens
    );

    // SECONDARY byte-equality proof that the cold output is reproduced.
    assert_eq!(
        r1.raw_text, r2.raw_text,
        "reset+rerun did not reproduce turn-1 raw_text byte-for-byte: \
         before={:?} after={:?}",
        r1.raw_text, r2.raw_text
    );
    assert_eq!(
        r1.num_tokens, r2.num_tokens,
        "reset+rerun did not reproduce turn-1 num_tokens: before={} after={}",
        r1.num_tokens, r2.num_tokens
    );
}
