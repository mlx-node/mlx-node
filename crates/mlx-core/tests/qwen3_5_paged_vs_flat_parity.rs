//! Real-weights paged-vs-flat numerical-equivalence gate for Qwen3.5
//! dense.
//!
//! Mirrors `lfm2_paged_vs_flat_parity.rs`. Qwen3.5's hybrid layer mix
//! (GDN linear-attention + full-attention) means only the
//! full-attention layers route through the paged adapter. Both sides of
//! the comparison are pure-Rust eager forwards (the compiled C++
//! qwen3.5 forward no longer exists): this gate verifies that the paged
//! forward is byte-equal to the flat forward (for greedy decoding) on
//! real weights.
//!
//! Gated on `MLX_TEST_MODEL_PATH` so a plain `cargo test --ignored`
//! without the env var still passes (the early-return short-circuits
//! the body before any model load).
//!
//! Run locally with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/qwen3_5-0.8b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_5_paged_vs_flat_parity \
//!     -- --ignored --nocapture
//! ```
//!
//! ⚠️ Until the dispatch is bit-correct on real weights this test
//! WILL FAIL. The infrastructure (early-return when env var unset,
//! flat / paged config patching) is what's gated to compile + skip
//! cleanly without weights.

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
use mlx_core::tokenizer::ChatMessage;

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

    let dst = workspace_target.join(format!("paged-parity-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    // Symlink weight files; only config.json mutated. Avoids disk-OOM.
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

    if use_block_paged {
        let cfg_path = dst.join("config.json");
        let raw = fs::read_to_string(&cfg_path)
            .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
        let mut cfg: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
        cfg["use_block_paged_cache"] = serde_json::Value::Bool(true);
        cfg["paged_cache_memory_mb"] = serde_json::Value::from(512u32);
        cfg["paged_block_size"] = serde_json::Value::from(16u32);
        let pretty = serde_json::to_string_pretty(&cfg)
            .map_err(|e| format!("serialize config.json: {e}"))?;
        fs::write(&cfg_path, pretty)
            .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;
    }

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
             ./.cache/models/qwen3_5-0.8b-mlx-bf16)"
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 dense checkpoint"]
async fn qwen3_5_paged_vs_flat_greedy_token_parity() {
    let Some(src) = resolve_source_model() else {
        return;
    };

    let flat_dir = match clone_model_dir(&src, "qwen35-flat", false) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for flat path: {e}"),
    };
    let paged_dir = match clone_model_dir(&src, "qwen35-paged", true) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for paged path: {e}"),
    };

    let flat_model = Qwen3_5Model::load(flat_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load flat-path Qwen3.5 model");
    let paged_model = Qwen3_5Model::load(paged_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load paged-path Qwen3.5 model");

    let prompts = parity_prompts();
    for (idx, prompt) in prompts.iter().enumerate() {
        let cfg_flat = parity_chat_config(32);
        let cfg_paged = parity_chat_config(32);

        let r_flat = flat_model
            .chat_session_start(vec![user_message(prompt)], Some(cfg_flat))
            .await
            .unwrap_or_else(|e| panic!("flat chat_session_start failed (#{idx}): {e:?}"));
        let r_paged = paged_model
            .chat_session_start(vec![user_message(prompt)], Some(cfg_paged))
            .await
            .unwrap_or_else(|e| panic!("paged chat_session_start failed (#{idx}): {e:?}"));

        eprintln!(
            "prompt #{idx} ({prompt:?}): flat num_tokens={} paged num_tokens={}",
            r_flat.num_tokens, r_paged.num_tokens
        );

        assert_eq!(
            r_flat.text, r_paged.text,
            "TEXT MISMATCH on prompt #{idx} ({prompt:?})\n\
             FLAT  text={:?}\nPAGED text={:?}",
            r_flat.text, r_paged.text,
        );
        assert_eq!(
            r_flat.num_tokens, r_paged.num_tokens,
            "num_tokens mismatch on prompt #{idx}",
        );
    }
    eprintln!(
        "Qwen3.5 dense greedy parity: all {} prompts matched",
        prompts.len()
    );
}
