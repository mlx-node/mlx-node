//! Correctness gate for the Qwen3.5 MoE **vision** (image+text) PAGED path.
//!
//! This is a CORRECTNESS gate, NOT a byte-exact-vs-flat parity gate. Paged
//! decode is intentionally ~1 bf16 ULP off from flat over long KV context: the
//! paged block-attention kernel reduces in a different order than flat's
//! monolithic SDPA, so the two diverge only at a late near-tie. (This is the
//! accepted paged-vs-eager gap; vLLM ships the identical thing and never forces
//! bit-equality.) An image is a long prefill, so a byte-exact-vs-flat VLM
//! assertion would be the WRONG bar — stricter than even paged-TEXT meets, and
//! stricter than vLLM holds itself to.
//!
//! Instead, matching the philosophy of `qwen3_5_moe_vl_image_chat.rs`, this
//! gate proves the paged vision path is CORRECT via three independent
//! properties:
//!   * DETERMINISM — paged(image) at T=0 is byte-identical run-to-run.
//!   * IMAGE-DEPENDENCE — paged(image) differs from paged(no-image), so the
//!     vision features actually reach generation (a path that silently dropped
//!     the image would fail this).
//!   * TRACKS-FLAT — paged(image) shares a long common prefix with flat(image).
//!     The prefill is bit-identical, so both paths start identically; a real
//!     M-RoPE/position bug would diverge almost immediately (well under 24
//!     chars), while the benign ~1-ULP decode tie only flips a near-tie much
//!     later. A >=24-char shared prefix therefore passes for a correct port and
//!     fails for a real bug — without demanding full byte-equality.
//!
//! The single source checkpoint is cloned twice (config-only patch:
//! `use_block_paged_cache` off vs on) so flat and paged differ only in cache
//! topology — every weight tensor is the same file (symlinked).
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
    }
}

const PROMPT: &str = "Describe this image briefly.";

/// Length (in chars) of the leading common prefix shared by `a` and `b`.
fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

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

    let flat_dir =
        clone_model_dir(&src, "qwen35moe-vlm-flat", false).expect("clone flat model dir failed");
    let paged_dir =
        clone_model_dir(&src, "qwen35moe-vlm-paged", true).expect("clone paged model dir failed");

    let flat_model = Qwen3_5MoeModel::load(flat_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load flat-path Qwen3.5-MoE-VL model");
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

    // --- 4. TRACKS-FLAT: paged(image) shares a long common prefix with
    // flat(image). The prefill is bit-identical, so the paths start identically;
    // a real M-RoPE/position bug would diverge almost immediately (well under
    // 24 chars), while the benign ~1-ULP paged-decode tie only flips a near-tie
    // much later. A >=24-char shared prefix therefore passes for a correct port
    // and fails for a real bug — without demanding full byte-equality. ---
    let flat_a = flat_model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image)],
            Some(correctness_chat_config(64)),
        )
        .await
        .expect("flat(image) chat_session_start failed");
    let shared = common_prefix_chars(&flat_a.text, &paged_a.text);
    eprintln!(
        "tracks-flat: flat num_tokens={} paged num_tokens={} common_prefix_chars={}",
        flat_a.num_tokens, paged_a.num_tokens, shared
    );
    assert!(
        shared >= 24,
        "paged(image) does not track flat(image): common prefix only {shared} chars \
         (a real M-RoPE/position bug diverges well under 24)\n\
         FLAT  text={:?}\nPAGED text={:?}",
        flat_a.text,
        paged_a.text,
    );

    eprintln!(
        "Qwen3.5-MoE-VL paged-VLM correctness: coherence + determinism + \
         image-dependence + tracks-flat ({shared}-char shared prefix) all passed"
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
