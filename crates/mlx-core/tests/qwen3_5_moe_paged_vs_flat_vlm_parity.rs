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
use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::{DynamicImage, GenericImageView, ImageFormat};
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
        cache_owner_id: None,
        cache_root_owner_id: None,
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
        thinking_enabled: None,
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
        thinking_enabled: None,
        images: None,
        audio: None,
    }
}

fn assistant_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
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

fn cache_lifecycle_chat_config() -> ChatConfig {
    let mut config = correctness_chat_config(16);
    // Full-history replay reconstructs prior assistant turns from the public
    // result. Disable hidden reasoning so that text contains every content
    // token needed to reproduce the committed prompt prefix exactly.
    config.thinking_token_budget = Some(0);
    config.include_reasoning = Some(false);
    config
}

fn cache_lifecycle_chat_config_for_owner(owner: &str) -> ChatConfig {
    let mut config = cache_lifecycle_chat_config();
    config.cache_owner_id = Some(owner.to_owned());
    config.cache_root_owner_id = Some("qwen35-moe-vlm-lifecycle-root".to_owned());
    config
}

/// Change the image content without changing its decoded dimensions. Both
/// prompts therefore expand to the same token shape, isolating the per-image
/// cache identity as the only reason for cache-chain backoff.
fn same_shape_image_variant(bytes: &[u8]) -> Vec<u8> {
    let original = image::load_from_memory(bytes).expect("test image must decode");
    let original_dimensions = original.dimensions();
    let mut rgba = original.to_rgba8();
    let pixel = rgba.get_pixel_mut(0, 0);
    pixel.0[0] = pixel.0[0].wrapping_add(1);

    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(rgba)
        .write_to(&mut encoded, ImageFormat::Png)
        .expect("encode same-shape image variant");
    let changed = encoded.into_inner();
    let changed_dimensions = image::load_from_memory(&changed)
        .expect("encoded image variant must decode")
        .dimensions();
    assert_eq!(original_dimensions, changed_dimensions);
    assert_ne!(bytes, changed.as_slice());
    changed
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

/// MoE twin of the dense image-aware paged-prefix lifecycle regression.
/// It follows the public stateless-agent/full-history flow and pins cold start,
/// owner A -> owner B -> owner A sidecar restoration, same-image reuse,
/// image-identity retention across a text continuation, and same-token-shape
/// cache backoff after changing only the image bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_QWEN35MOE_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH"]
async fn qwen3_5_moe_paged_vlm_image_prefix_cache_lifecycle() {
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
    let image_a = std::fs::read(&image_path).expect("failed to read test image");
    let image_b = same_shape_image_variant(&image_a);

    let paged_dir = clone_model_dir(&src, "qwen35moe-vlm-image-prefix-cache", true)
        .expect("clone paged model dir failed");
    let model = Qwen3_5MoeModel::load(paged_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load paged-path Qwen3.5-MoE-VL model");

    // 1. No live or published blocks exist on a newly loaded model.
    let first = model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image_a)],
            Some(cache_lifecycle_chat_config_for_owner("owner-a")),
        )
        .await
        .expect("first image turn failed");
    assert_eq!(
        first.cached_tokens, 0,
        "first image turn must cold-prefill: {first:?}"
    );

    // Force the live adapter request and recurrent caches away from owner A.
    // Owner A's content-addressed K/V blocks and exact GDN sidecar must remain
    // available independently of owner B's active state.
    let owner_b = model
        .chat_session_start(
            vec![user_message_with_image(
                "State the dominant color in this image.",
                &image_b,
            )],
            Some(cache_lifecycle_chat_config_for_owner("owner-b")),
        )
        .await
        .expect("owner B displacement turn failed");
    assert!(
        owner_b.num_tokens > 0,
        "owner B produced no output: {owner_b:?}"
    );

    let owner_a_replay = model
        .chat_session_start(
            vec![user_message_with_image(PROMPT, &image_a)],
            Some(cache_lifecycle_chat_config_for_owner("owner-a")),
        )
        .await
        .expect("owner A same-image replay after owner B failed");
    assert_eq!(
        owner_a_replay.text, first.text,
        "owner displacement changed exact T=0 output"
    );
    assert_eq!(owner_a_replay.num_tokens, first.num_tokens);
    assert!(
        owner_a_replay.cached_tokens > 0,
        "owner A replay after owner B must restore an exact cached prefix: {owner_a_replay:?}"
    );

    // 2. Re-render the full committed transcript with the identical image.
    let followup = "Name one visible detail.";
    let same_history = vec![
        user_message_with_image(PROMPT, &image_a),
        assistant_message(&first.text),
        user_message(followup),
    ];
    let replay = model
        .chat_session_start(
            same_history,
            Some(cache_lifecycle_chat_config_for_owner("owner-a")),
        )
        .await
        .expect("same-image full-history replay failed");
    assert!(
        replay.cached_tokens > 0,
        "identical image full-history replay must reuse cached blocks: {replay:?}"
    );

    // 3. The paged text continuation must keep the original image hashes on
    // its re-published block chain.
    let continuation = "Now answer with a short noun.";
    let continued = model
        .chat_session_continue(
            continuation.to_string(),
            None,
            None,
            Some(cache_lifecycle_chat_config_for_owner("owner-a")),
        )
        .await
        .expect("text continuation after image replay failed");
    assert!(
        continued.cached_tokens > 0,
        "text continuation must reuse the image-bearing live prefix: {continued:?}"
    );

    let final_question = "Confirm in one word.";
    let extended_history_a = vec![
        user_message_with_image(PROMPT, &image_a),
        assistant_message(&first.text),
        user_message(followup),
        assistant_message(&replay.text),
        user_message(continuation),
        assistant_message(&continued.text),
        user_message(final_question),
    ];
    let after_text = model
        .chat_session_start(
            extended_history_a,
            Some(cache_lifecycle_chat_config_for_owner("owner-a")),
        )
        .await
        .expect("same-image replay after text continuation failed");
    assert!(
        after_text.cached_tokens > 0,
        "text continuation lost the image block identity: {after_text:?}"
    );

    // 4. Only the raw content changes; the image geometry and entire rendered
    // text history remain identical. Leading text-only blocks may still hit,
    // but reuse must stop at the first image-conditioned block.
    let extended_history_b = vec![
        user_message_with_image(PROMPT, &image_b),
        assistant_message(&first.text),
        user_message(followup),
        assistant_message(&replay.text),
        user_message(continuation),
        assistant_message(&continued.text),
        user_message(final_question),
    ];
    let changed = model
        .chat_session_start(
            extended_history_b,
            Some(cache_lifecycle_chat_config_for_owner("owner-a")),
        )
        .await
        .expect("changed-image full-history replay failed");
    assert!(
        changed.cached_tokens < after_text.cached_tokens,
        "same-shape changed image reused through the image-conditioned prefix: \
         same-image cached_tokens={} changed-image cached_tokens={} changed={changed:?}",
        after_text.cached_tokens,
        changed.cached_tokens,
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
