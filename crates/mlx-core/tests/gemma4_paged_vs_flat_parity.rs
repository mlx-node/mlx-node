//! Real-weights paged-vs-flat numerical-equivalence gate for Gemma4.
//!
//! Validates that the block-paged KV cache adapter (reached via
//! `chat_sync_core_paged`) produces byte-equal greedy-decode token output
//! to the legacy flat `Vec<Gemma4LayerCache>` path (reached via
//! `chat_sync_core`) when fed identical real-model weights and identical
//! prompts. This is the gate that has to be green before
//! `use_block_paged_cache`'s default flips from `Some(false)` to
//! `Some(true)` for Gemma4.
//!
//! Mirrors `qwen3_paged_vs_flat_parity.rs` and `lfm2_paged_vs_flat_parity.rs`.
//! Both paths load from the SAME on-disk Gemma4 checkpoint, with only
//! `use_block_paged_cache` patched in the paged copy's `config.json`.
//!
//! Gated on `MLX_TEST_MODEL_PATH` so a plain `cargo test --ignored` without
//! the env var still passes (the early-return short-circuits before any
//! model load). Gemma4's hybrid sliding+global attention, dual RoPE
//! variants, MoE/PLE, and KV-sharing make the paged forward dispatch
//! significantly more complex than Qwen3's, so this test is expected to
//! reveal divergences that subsequent commits will need to fix. It is
//! marked `#[ignore]` and DOES NOT block default CI.
//!
//! Run locally with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/gemma-3-1b-mlx \
//!     cargo test -p mlx-core --test gemma4_paged_vs_flat_parity \
//!     -- --ignored --nocapture
//! ```
//!
//! Pass criteria: byte-equal `text` and `num_tokens` between flat and
//! paged paths over 4 distinct prompts at temperature=0 / max_new_tokens=32
//! (test a) plus byte-equal across a two-turn dialog whose second turn
//! exercises the prefix-reuse machinery (test b).

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::tokenizer::ChatMessage;

// ---------------------------------------------------------------------------
// Test fixture helpers
// ---------------------------------------------------------------------------

/// Copy the source Gemma4 checkpoint directory into a fresh tempdir
/// (under `target/`) and optionally patch `config.json` to turn on the
/// block-paged adapter.
fn clone_model_dir(src: &Path, suffix: &str, use_block_paged: bool) -> Result<PathBuf, String> {
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
    let dst = workspace_target.join(format!("gemma4-paged-parity-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    // Symlink large weight files instead of copying them; the only file we
    // actually need to mutate per-clone is `config.json`. Avoids OOM on disk
    // when the source checkpoint is multi-GB.
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

    // Always explicitly pin `use_block_paged_cache` — the default flipped
    // to `true` once parity landed, so a missing key would silently route
    // BOTH "flat" and "paged" copies through the paged path and reduce
    // the parity test to a no-op (paged-vs-paged). Pin explicitly so the
    // test is unambiguous regardless of default-flip drift.
    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path)
        .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
    let mut cfg: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(use_block_paged);
    if use_block_paged {
        cfg["paged_cache_memory_mb"] = serde_json::Value::from(512u32);
        cfg["paged_block_size"] = serde_json::Value::from(16u32);
    }
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty)
        .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;

    Ok(dst)
}

fn parity_chat_config(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        min_p: None,
        repetition_penalty: Some(1.0),
        repetition_context_size: None,
        presence_penalty: Some(0.0),
        presence_context_size: None,
        frequency_penalty: Some(0.0),
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

fn parity_prompts() -> [&'static str; 4] {
    [
        "Say hi in one short word.",
        "What is 2 + 3? Answer with just the number.",
        "Name a primary color.",
        "Complete: the sky is",
    ]
}

fn resolve_source_model() -> Option<PathBuf> {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/gemma-3-1b-mlx)"
        );
        return None;
    };
    let p = PathBuf::from(&model_path);
    if !p.exists() {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH does not exist: {}",
            p.display()
        );
        return None;
    }
    Some(p)
}

// ---------------------------------------------------------------------------
// Test (a): greedy-decode token parity over 4 prompts
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_paged_vs_flat_greedy_token_parity() {
    use mlx_core::models::gemma4::model::Gemma4Model;

    let Some(src) = resolve_source_model() else {
        return;
    };

    let flat_dir = match clone_model_dir(&src, "gemma4-flat", false) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for flat path: {e}"),
    };
    let paged_dir = match clone_model_dir(&src, "gemma4-paged", true) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for paged path: {e}"),
    };

    let flat_model = Gemma4Model::load_from_dir(&flat_dir.to_string_lossy(), None)
        .await
        .expect("failed to load flat-path Gemma4 model");
    let paged_model = Gemma4Model::load_from_dir(&paged_dir.to_string_lossy(), None)
        .await
        .expect("failed to load paged-path Gemma4 model");

    let prompts = parity_prompts();
    for (idx, prompt) in prompts.iter().enumerate() {
        let r_flat = flat_model
            .chat_session_start(vec![user_message(prompt)], Some(parity_chat_config(32)))
            .await
            .unwrap_or_else(|e| panic!("flat chat_session_start failed (prompt #{idx}): {e:?}"));
        let r_paged = paged_model
            .chat_session_start(vec![user_message(prompt)], Some(parity_chat_config(32)))
            .await
            .unwrap_or_else(|e| panic!("paged chat_session_start failed (prompt #{idx}): {e:?}"));

        eprintln!(
            "prompt #{idx} ({prompt:?}): flat num_tokens={} paged num_tokens={} | \
             flat finish={} paged finish={}",
            r_flat.num_tokens, r_paged.num_tokens, r_flat.finish_reason, r_paged.finish_reason
        );

        if r_flat.text != r_paged.text {
            let first_diff = r_flat
                .text
                .as_bytes()
                .iter()
                .zip(r_paged.text.as_bytes().iter())
                .position(|(a, b)| a != b);
            panic!(
                "TEXT MISMATCH on prompt #{idx} ({prompt:?}). \
                 first_diff_byte={first_diff:?}\n\
                 FLAT  ({} tokens) text={:?}\n\
                 PAGED ({} tokens) text={:?}",
                r_flat.num_tokens, r_flat.text, r_paged.num_tokens, r_paged.text,
            );
        }
        assert_eq!(
            r_flat.num_tokens, r_paged.num_tokens,
            "num_tokens mismatch on prompt #{idx} ({prompt:?}): flat={} paged={}",
            r_flat.num_tokens, r_paged.num_tokens,
        );
        assert_eq!(
            r_flat.finish_reason, r_paged.finish_reason,
            "finish_reason mismatch on prompt #{idx}: flat={} paged={}",
            r_flat.finish_reason, r_paged.finish_reason,
        );
    }

    eprintln!(
        "Gemma4 greedy parity: all {} prompts matched",
        prompts.len()
    );
}

// ---------------------------------------------------------------------------
// Test (b): two-turn dialog parity (exercises prefix-reuse)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_paged_vs_flat_prefix_reuse_parity() {
    use mlx_core::models::gemma4::model::Gemma4Model;

    let Some(src) = resolve_source_model() else {
        return;
    };

    let flat_dir = match clone_model_dir(&src, "gemma4-flat-2t", false) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for flat path: {e}"),
    };
    let paged_dir = match clone_model_dir(&src, "gemma4-paged-2t", true) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for paged path: {e}"),
    };

    let flat_model = Gemma4Model::load_from_dir(&flat_dir.to_string_lossy(), None)
        .await
        .expect("failed to load flat-path Gemma4 model");
    let paged_model = Gemma4Model::load_from_dir(&paged_dir.to_string_lossy(), None)
        .await
        .expect("failed to load paged-path Gemma4 model");

    let prompt1 = "Say hi in one short word.";
    let r1_flat = flat_model
        .chat_session_start(vec![user_message(prompt1)], Some(parity_chat_config(32)))
        .await
        .expect("turn 1 flat chat_session_start failed");
    let r1_paged = paged_model
        .chat_session_start(vec![user_message(prompt1)], Some(parity_chat_config(32)))
        .await
        .expect("turn 1 paged chat_session_start failed");

    assert_eq!(
        r1_flat.text, r1_paged.text,
        "turn 1 text mismatch: flat={:?} paged={:?}",
        r1_flat.text, r1_paged.text
    );
    assert_eq!(
        r1_flat.num_tokens, r1_paged.num_tokens,
        "turn 1 num_tokens mismatch: flat={} paged={}",
        r1_flat.num_tokens, r1_paged.num_tokens
    );

    let user2 = "And in another word?";
    let r2_flat = flat_model
        .chat_session_continue(user2.to_string(), None, None, Some(parity_chat_config(32)))
        .await
        .expect("turn 2 flat chat_session_continue failed");
    let r2_paged = paged_model
        .chat_session_continue(user2.to_string(), None, None, Some(parity_chat_config(32)))
        .await
        .expect("turn 2 paged chat_session_continue failed");

    eprintln!(
        "two-turn parity: turn2 flat num_tokens={} cached={} | paged num_tokens={} cached={}",
        r2_flat.num_tokens, r2_flat.cached_tokens, r2_paged.num_tokens, r2_paged.cached_tokens,
    );

    if r2_flat.raw_text != r2_paged.raw_text {
        panic!(
            "TURN-2 RAW_TEXT MISMATCH (prefix-reuse divergence between paths)\n\
             FLAT  ({} tokens, cached={}) raw_text={:?}\n\
             PAGED ({} tokens, cached={}) raw_text={:?}",
            r2_flat.num_tokens,
            r2_flat.cached_tokens,
            r2_flat.raw_text,
            r2_paged.num_tokens,
            r2_paged.cached_tokens,
            r2_paged.raw_text,
        );
    }
    assert_eq!(
        r2_flat.num_tokens, r2_paged.num_tokens,
        "turn 2 num_tokens mismatch (prefix-reuse divergence): flat={} paged={}",
        r2_flat.num_tokens, r2_paged.num_tokens,
    );
}
