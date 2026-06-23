//! Correctness gate for the Gemma4 **vision** (image+text) PAGED path.
//!
//! Gemma4 image turns run ONLY on the block-paged KV backend; there is no
//! flat-vision path. This gate therefore proves the paged vision path is
//! CORRECT on its own terms (no flat reference exists to compare against) via
//! three independent properties, matching the philosophy of
//! `gemma4_vl_image_chat.rs`:
//!   * COHERENCE — paged(image) produces real (non-empty) output.
//!   * DETERMINISM — paged(image) at T=0 is byte-identical run-to-run.
//!   * IMAGE-DEPENDENCE — paged(image) differs from paged(no-image), so the
//!     vision features actually reach generation (a path that silently dropped
//!     the image would fail this).
//!
//! Plus an ERROR-CONTRACT property: a model loaded WITHOUT a paged adapter
//! (`use_block_paged_cache: false`) has no vision path, so an image turn must
//! return an error indicating the block-paged backend is required — it must
//! NOT silently fall back.
//!
//! The single source checkpoint is cloned with a config-only patch
//! (`use_block_paged_cache` on for the correctness clones, off for the
//! error-contract clone) so the clones differ only in cache topology — every
//! weight tensor is the same file (symlinked).
//!
//! Gated on `MLX_TEST_GEMMA4_VL_MODEL_PATH` (a Gemma-4-VL checkpoint, e.g.
//! gemma-4-e2b-it-mlx) and a test image (`MLX_TEST_VLM_IMAGE_PATH` else
//! `examples/ocr.png`). A plain `cargo test --ignored` without the env vars
//! early-returns before any model load, so it passes cleanly.
//!
//! Run locally with:
//!
//! ```shell
//! MLX_TEST_GEMMA4_VL_MODEL_PATH=./.cache/models/gemma-4-e2b-it-mlx \
//!     MLX_TEST_VLM_IMAGE_PATH=examples/ocr.png \
//!     cargo test -p mlx-core --test gemma4_paged_vs_flat_vlm_parity \
//!     -- --ignored --nocapture
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::gemma4::model::Gemma4Model;
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

    let dst = workspace_target.join(format!("gemma4-vlm-correctness-{pid}-{suffix}"));
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

    // Always explicitly pin `use_block_paged_cache` — the default is `true`,
    // so a missing key in the "flat" copy would silently route BOTH copies
    // through the paged path and reduce the test to paged-vs-paged.
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
#[ignore = "needs MLX_TEST_GEMMA4_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH"]
async fn gemma4_paged_vlm_correctness() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_VL_MODEL_PATH unset");
        return;
    };
    let src = PathBuf::from(&model_path);
    if !src.exists() {
        eprintln!(
            "skipping: MLX_TEST_GEMMA4_VL_MODEL_PATH does not exist: {}",
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
        clone_model_dir(&src, "gemma4-vlm-paged", true).expect("clone paged model dir failed");

    let paged_model = Gemma4Model::load(paged_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load paged-path Gemma-4-VL model");

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
        "Gemma-4-VL paged-VLM correctness: coherence + determinism + \
         image-dependence all passed (paged num_tokens={})",
        paged_a.num_tokens
    );
}

/// Regression: a paged IMAGE turn must leave the session in gemma4's
/// image-held state so a following text-only `chat_session_continue` is
/// REJECTED with the restart-required prefix — matching gemma4's FLAT contract.
///
/// Unlike qwen3.5 (which preserves image context across a continue), gemma4's
/// prefix reuse is text-only: a text delta on top of an image session would
/// prefill on caches whose positions include expanded image tokens the history
/// bookkeeping does not model. So gemma4's `text_delta_image_guard` rejects
/// (keyed on `cached_image_key.is_some()`). The paged image core MUST arm that
/// key (`cached_image_key = Some(...)`) at the end; otherwise the guard would
/// not fire and the continue would silently drop the image. This locks the
/// reject in for the paged path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_GEMMA4_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH"]
async fn gemma4_paged_vlm_continue_is_rejected() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_VL_MODEL_PATH unset");
        return;
    };
    let src = PathBuf::from(&model_path);
    if !src.exists() {
        eprintln!(
            "skipping: MLX_TEST_GEMMA4_VL_MODEL_PATH does not exist: {}",
            src.display()
        );
        return;
    }
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    let paged_dir = clone_model_dir(&src, "gemma4-vlm-paged-continue", true)
        .expect("clone paged model dir failed");
    let paged_model = Gemma4Model::load(paged_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load paged-path Gemma-4-VL model");

    // Turn 1: paged image turn.
    let r1 = paged_model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image)],
            Some(correctness_chat_config(48)),
        )
        .await
        .expect("paged(image) chat_session_start failed");
    assert!(r1.num_tokens > 0, "image turn produced zero tokens: {r1:?}");

    // Turn 2: text-only continue must be REJECTED with the restart-required
    // prefix (gemma4 keeps `cached_image_key` armed so the guard fires).
    let cont = paged_model
        .chat_session_continue(
            "Answer in one word: is there text?".to_string(),
            None,
            None,
            Some(correctness_chat_config(48)),
        )
        .await;
    let err = cont.expect_err(
        "gemma4 must REJECT a text continue on a paged image session (matching its flat contract)",
    );
    assert!(
        err.reason.contains("IMAGE_CHANGE_REQUIRES_SESSION_RESTART"),
        "expected image-session restart rejection, got: {}",
        err.reason
    );

    eprintln!(
        "Gemma-4-VL paged-VLM continue: text continue correctly REJECTED \
         (image session demands restart)"
    );
}

/// Error contract: a model loaded WITHOUT a paged adapter
/// (`use_block_paged_cache: false`) has no vision path. An image turn must
/// return an error indicating the block-paged backend is required — it must
/// NOT silently fall back to a flat path (the flat-vision path was removed).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_GEMMA4_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH"]
async fn gemma4_image_turn_without_paged_adapter_errors() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_VL_MODEL_PATH unset");
        return;
    };
    let src = PathBuf::from(&model_path);
    if !src.exists() {
        eprintln!(
            "skipping: MLX_TEST_GEMMA4_VL_MODEL_PATH does not exist: {}",
            src.display()
        );
        return;
    }
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    // Clone with `use_block_paged_cache: false` so the model loads with no
    // paged adapter (the only configuration where the vision path is absent).
    let flat_dir = clone_model_dir(&src, "gemma4-vlm-no-paged", false)
        .expect("clone no-paged model dir failed");
    let flat_model = Gemma4Model::load(flat_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load no-paged Gemma-4-VL model");

    let result = flat_model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image)],
            Some(correctness_chat_config(48)),
        )
        .await;
    let err = result.expect_err(
        "an image turn on a model with no paged adapter must error, not silently fall back",
    );
    assert!(
        err.reason.contains("block-paged"),
        "expected a 'block-paged backend required' error, got: {}",
        err.reason
    );

    eprintln!(
        "Gemma-4-VL no-paged image turn: correctly ERRORED \
         (block-paged backend required): {}",
        err.reason
    );
}
