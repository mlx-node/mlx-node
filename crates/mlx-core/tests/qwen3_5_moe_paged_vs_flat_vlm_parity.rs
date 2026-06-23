//! Correctness + error-contract gate for the Qwen3.5 MoE **vision**
//! (image+text) path.
//!
//! MoE image turns run ONLY on the block-paged backend. VLM checkpoints default
//! to paged at load; the flat path no longer has a vision arm, so a vision turn
//! that reaches a None paged adapter ERRORS at dispatch. This file therefore
//! proves two things:
//!
//!   1. The paged vision path is CORRECT (the only path image turns take).
//!   2. A flat-loaded (`use_block_paged_cache: false`) clone REJECTS an image
//!      turn with a clear "requires the block-paged KV backend" error, rather
//!      than silently running a removed flat-vision path.
//!
//! For (1) this is a CORRECTNESS gate, NOT a byte-exact-vs-flat parity gate
//! (the flat-vision path is gone, so there is nothing to compare against).
//! Matching the philosophy of `qwen3_5_moe_vl_image_chat.rs`, it proves the
//! paged vision path is CORRECT via three independent properties:
//!   * COHERENCE — paged(image) produces real (non-empty) output.
//!   * DETERMINISM — paged(image) at T=0 is byte-identical run-to-run.
//!   * IMAGE-DEPENDENCE — paged(image) differs from paged(no-image), so the
//!     vision features actually reach generation (a path that silently dropped
//!     the image would fail this).
//!
//! The source checkpoint is cloned with a config-only patch
//! (`use_block_paged_cache` on for the paged clone, off for the error-contract
//! clone) so the clones differ only in cache topology — every weight tensor is
//! the same file (symlinked).
//!
//! Gated on `MLX_TEST_QWEN35MOE_VL_MODEL_PATH` (a MoE vision checkpoint) and a
//! test image (`MLX_TEST_VLM_IMAGE_PATH` else `examples/ocr.png`). A plain
//! `cargo test --ignored` without the env vars early-returns before any model
//! load, so it passes cleanly.
//!
//! Run locally with:
//!
//! ```shell
//! MLX_TEST_QWEN35MOE_VL_MODEL_PATH=./.cache/models/Qwen3.6-35b-a3b-mlx \
//!     MLX_TEST_VLM_IMAGE_PATH=examples/ocr.png \
//!     cargo test -p mlx-core --test qwen3_5_moe_paged_vs_flat_vlm_parity \
//!     -- --ignored --nocapture
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use mlx_core::tokenizer::ChatMessage;
use napi::bindgen_prelude::Uint8Array;

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

    let dst = workspace_target.join(format!("paged-moe-vlm-correctness-{pid}-{suffix}"));
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

    // Always explicitly pin `use_block_paged_cache` (mirrors the gemma4
    // helper). A conditional write on the flat copy would silently route BOTH
    // copies through the paged path if the loader default flipped to `true` or
    // the source config gained the key — collapsing the gate to paged-vs-paged.
    // The memory/block knobs only matter for the paged copy.
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

fn correctness_chat_config(max_new_tokens: i32) -> ChatConfig {
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

fn user_message_with_image(content: &str, image: &[u8]) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        images: Some(vec![Uint8Array::new(image.to_vec())]),
        audio: None,
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

const PROMPT: &str = "Describe this image briefly.";

/// Resolve the test image: `MLX_TEST_VLM_IMAGE_PATH` else `examples/ocr.png`
/// relative to the repo root (CARGO_MANIFEST_DIR is `crates/mlx-core`, so the
/// repo root is two levels up).
fn resolve_image_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MLX_TEST_VLM_IMAGE_PATH") {
        let pb = PathBuf::from(p);
        return pb.exists().then_some(pb);
    }
    let pb = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/ocr.png");
    pb.exists().then_some(pb)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_QWEN35MOE_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH"]
async fn qwen3_5_moe_paged_vlm_correctness() {
    let Ok(model_path) = std::env::var("MLX_TEST_QWEN35MOE_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_QWEN35MOE_VL_MODEL_PATH unset");
        return;
    };
    let src = PathBuf::from(&model_path);
    if !src.exists() {
        eprintln!(
            "skipping: MLX_TEST_QWEN35MOE_VL_MODEL_PATH does not exist: {}",
            src.display()
        );
        return;
    }
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    let paged_dir =
        clone_model_dir(&src, "qwen35moe-vlm-paged", true).expect("clone paged model dir failed");

    let paged_model = Qwen3_5MoeModel::load(paged_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load paged-path Qwen3.5-MoE-VL model");

    // --- 1. COHERENCE: paged(image) produces real output. ---
    let paged_a = paged_model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image)],
            Some(correctness_chat_config(64)),
        )
        .await
        .expect("paged(image) chat_session_start failed");
    assert!(
        paged_a.num_tokens > 0,
        "paged(image) produced zero tokens: {paged_a:?}"
    );

    // --- 2. DETERMINISM: paged(image) at T=0 is byte-identical run-to-run. ---
    tokio::task::block_in_place(|| paged_model.reset_caches()).expect("reset_caches failed");
    let paged_b = paged_model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image)],
            Some(correctness_chat_config(64)),
        )
        .await
        .expect("paged(image) re-run chat_session_start failed");
    assert_eq!(
        paged_a.text, paged_b.text,
        "paged(image) is not deterministic at T=0:\nrun A={:?}\nrun B={:?}",
        paged_a.text, paged_b.text,
    );
    assert_eq!(
        paged_a.num_tokens, paged_b.num_tokens,
        "paged(image) num_tokens not deterministic at T=0",
    );

    // --- 3. IMAGE-DEPENDENCE: paged(image) differs from paged(no-image). ---
    tokio::task::block_in_place(|| paged_model.reset_caches()).expect("reset_caches failed");
    let paged_no_image = paged_model
        .chat_session_start(
            vec![user_message(PROMPT)],
            Some(correctness_chat_config(64)),
        )
        .await
        .expect("paged(no-image) chat_session_start failed");
    assert_ne!(
        paged_a.text, paged_no_image.text,
        "paged path ignored the image (with/without image produced identical output)"
    );

    eprintln!(
        "Qwen3.5-MoE-VL paged-VLM correctness: coherence + determinism + \
         image-dependence all passed"
    );
}

/// Regression: a paged IMAGE turn must leave a CONTINUABLE session that
/// preserves the image context.
///
/// MoE qwen3.5 `supports_images() == true`, so a text-only
/// `chat_session_continue` after an image turn is ACCEPTED. The image turn
/// MUST keep its paged blocks live + save the expanded history so the continue
/// extends the live image-bearing KV instead of rebuilding from an empty
/// history (which would silently DROP the image and prior turn). Proven by
/// `cached_tokens > 0` on the continue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_QWEN35MOE_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH"]
async fn qwen3_5_moe_paged_vlm_continue_preserves_image_context() {
    let Ok(model_path) = std::env::var("MLX_TEST_QWEN35MOE_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_QWEN35MOE_VL_MODEL_PATH unset");
        return;
    };
    let src = PathBuf::from(&model_path);
    if !src.exists() {
        eprintln!(
            "skipping: MLX_TEST_QWEN35MOE_VL_MODEL_PATH does not exist: {}",
            src.display()
        );
        return;
    }
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    let paged_dir = clone_model_dir(&src, "qwen35moe-vlm-paged-continue", true)
        .expect("clone paged model dir failed");
    let paged_model = Qwen3_5MoeModel::load(paged_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load paged-path Qwen3.5-MoE-VL model");

    // Turn 1: paged image turn.
    let r1 = paged_model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image)],
            Some(correctness_chat_config(48)),
        )
        .await
        .expect("paged(image) chat_session_start failed");
    assert!(r1.num_tokens > 0, "image turn produced zero tokens: {r1:?}");

    // Turn 2: text-only continue referencing the image. Must be accepted AND
    // reuse the saved image-expanded prefix (cached_tokens > 0).
    let r2 = paged_model
        .chat_session_continue(
            "Answer in one word: what is in the image?".to_string(),
            None,
            None,
            Some(correctness_chat_config(48)),
        )
        .await
        .expect("text continue after paged image turn must be ACCEPTED, not error");

    eprintln!(
        "continue-preserves-image: turn1 num_tokens={} | turn2 num_tokens={} cached_tokens={} prompt_tokens={}",
        r1.num_tokens, r2.num_tokens, r2.cached_tokens, r2.prompt_tokens,
    );

    assert!(
        r2.cached_tokens > 0,
        "continue after paged image turn DROPPED the image context (cached_tokens=0): the \
         paged image turn did not keep its blocks live / save history. \
         turn2={r2:?}"
    );
    assert!(
        r2.num_tokens > 0,
        "continue after paged image turn produced zero tokens: {r2:?}"
    );

    eprintln!(
        "Qwen3.5-MoE-VL paged-VLM continue: image context preserved \
         (cached_tokens={} > 0)",
        r2.cached_tokens
    );
}

/// Error contract: a VLM checkpoint loaded with `use_block_paged_cache: false`
/// has NO paged adapter, so an image turn must ERROR (the flat-vision path was
/// removed) rather than silently running text-only or crashing. The message
/// must indicate the block-paged backend is required.
///
/// MoE has no paged-override env, so the explicit `false` is the only thing
/// that decides cache topology here: it survives the vision->paged load-force,
/// leaving the clone flat.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_QWEN35MOE_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH"]
async fn qwen3_5_moe_flat_vlm_image_turn_errors_without_paged_backend() {
    let Ok(model_path) = std::env::var("MLX_TEST_QWEN35MOE_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_QWEN35MOE_VL_MODEL_PATH unset");
        return;
    };
    let src = PathBuf::from(&model_path);
    if !src.exists() {
        eprintln!(
            "skipping: MLX_TEST_QWEN35MOE_VL_MODEL_PATH does not exist: {}",
            src.display()
        );
        return;
    }
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    // Clone with `use_block_paged_cache: false` — this explicit false survives
    // the vision->paged load-force, so no paged adapter is built.
    let flat_dir = clone_model_dir(&src, "qwen35moe-vlm-flat-error", false)
        .expect("clone flat model dir failed");
    let flat_model = Qwen3_5MoeModel::load(flat_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load flat-path Qwen3.5-MoE-VL model");

    let result = flat_model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image)],
            Some(correctness_chat_config(16)),
        )
        .await;

    let err = result.expect_err(
        "flat VLM image turn must ERROR (no paged adapter, flat-vision path removed), \
         not produce a ChatResult",
    );
    let msg = err.to_string();
    eprintln!("flat-VLM image-turn error message: {msg}");
    assert!(
        msg.contains("block-paged"),
        "flat VLM image-turn error must indicate the block-paged backend is required; got: {msg}"
    );
}
