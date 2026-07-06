//! Gemma-4-VL image-chat byte-identity gate for the vision merge-contract
//! genericization (S0 golden).
//!
//! Gemma4 is the DISJOINT vision family: it merges image features via
//! `masked_scatter` + a 4D attention mask and has NO M-RoPE / cross-turn
//! rope-delta (unlike the qwen3.5 dense+moe pair). It is the S4 decision
//! checkpoint — the expectation is that it does NOT fit the qwen3.5 M-RoPE
//! merge contract and stays on its own core. This gate exists so that decision
//! is made against a real byte-identity baseline: whatever S1–S3 do to the
//! shared engine, the gemma image-chat path must stay BYTE-IDENTICAL.
//!
//! Unlike the qwen gates, gemma is captured with a SINGLE image turn: gemma's
//! continue path refuses a text delta on an image session
//! (`IMAGE_CHANGE_REQUIRES_SESSION_RESTART: ... session currently holds image
//! state`) precisely because it has no rope-delta machinery to re-anchor a
//! text delta onto an image KV prefix. That restart-required behavior is itself
//! a load-bearing part of gemma's contract, so the gate asserts it directly.
//! Also asserts image-dependence (no-image control must differ) and
//! determinism (replay byte-identical).
//!
//! Gated on a real Gemma-4-VL checkpoint + a test image. Use a high-fidelity
//! checkpoint: low-bit quants (e.g. the A4B Q3_K_XL) emit `<pad>` garbage for
//! image inputs, so they are not a usable vision golden. Run:
//!   MLX_TEST_GEMMA4_VL_MODEL_PATH=.cache/models/gemma-4-e2b-it-mlx \
//!   MLX_TEST_VLM_IMAGE_PATH=examples/ocr.png \
//!     cargo test -p mlx-core --test gemma4_vl_image_chat -- --ignored --nocapture

use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::gemma4::model::Gemma4Model;
use mlx_core::tokenizer::ChatMessage;
use napi::bindgen_prelude::Uint8Array;

fn cfg(max_new_tokens: i32) -> ChatConfig {
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
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: None,
        mtp_depth: None,
        mtp_adaptive_depth: None,
    }
}

fn user_msg(content: &str, image: Option<&[u8]>) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        images: image.map(|b| vec![Uint8Array::new(b.to_vec())]),
        audio: None,
    }
}

fn resolve_image_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MLX_TEST_VLM_IMAGE_PATH") {
        let pb = PathBuf::from(p);
        return pb.exists().then_some(pb);
    }
    let pb = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/ocr.png");
    pb.exists().then_some(pb)
}

fn hash8(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn oneline(s: &str) -> String {
    s.replace('\n', "\\n").chars().take(72).collect()
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Digest {
    num_tokens: u32,
    cached_tokens: u32,
    finish_reason: String,
    raw_hash: u64,
}

async fn run_image_turn(m: &Gemma4Model, image: &[u8]) -> (Digest, String) {
    let t1 = m
        .chat_session_start(
            vec![user_msg("Describe this image briefly.", Some(image))],
            Some(cfg(48)),
        )
        .await
        .expect("image chat_session_start failed");
    let d1 = Digest {
        num_tokens: t1.num_tokens,
        cached_tokens: t1.cached_tokens,
        finish_reason: t1.finish_reason.clone(),
        raw_hash: hash8(&t1.raw_text),
    };
    (d1, t1.raw_text)
}

async fn describe_without_image(m: &Gemma4Model) -> Digest {
    let t = m
        .chat_session_start(
            vec![user_msg("Describe this image briefly.", None)],
            Some(cfg(48)),
        )
        .await
        .expect("no-image control chat_session_start failed");
    Digest {
        num_tokens: t.num_tokens,
        cached_tokens: t.cached_tokens,
        finish_reason: t.finish_reason.clone(),
        raw_hash: hash8(&t.raw_text),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH for a Gemma-4-VL checkpoint + test image"]
async fn gemma4_vl_image_chat_t0_capture() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_VL_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_GEMMA4_VL_MODEL_PATH does not exist: {model_path}"
    );
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    let model = Gemma4Model::load(model_path.clone(), None)
        .await
        .expect("failed to load Gemma-4-VL model");

    let reset = |m: &Gemma4Model| {
        tokio::task::block_in_place(|| m.reset_caches()).expect("reset_caches failed");
    };

    // --- Pass 1: capture the image-turn digest. ---
    reset(&model);
    let (d1, raw1) = run_image_turn(&model, &image).await;

    println!(
        "DIGEST image ntok={} cached={} finish={} sha={:016x} :: {}",
        d1.num_tokens,
        d1.cached_tokens,
        d1.finish_reason,
        d1.raw_hash,
        oneline(&raw1)
    );

    assert!(d1.num_tokens > 0, "image turn produced zero tokens: {d1:?}");
    assert!(
        d1.finish_reason == "stop" || d1.finish_reason == "length",
        "image turn unexpected finish_reason: {}",
        d1.finish_reason
    );

    // Gemma contract: a continue on an image session is REJECTED (no rope-delta
    // to re-anchor a text delta onto the image KV prefix), so it demands a
    // session restart. Lock that in — even a text-only continue must error.
    let cont = model
        .chat_session_continue(
            "Answer in one word: is there text?".to_string(),
            None,
            None,
            Some(cfg(48)),
        )
        .await;
    let err = cont.expect_err("gemma must reject continue on an image session");
    assert!(
        err.reason.contains("IMAGE_CHANGE_REQUIRES_SESSION_RESTART"),
        "expected image-session restart rejection, got: {}",
        err.reason
    );

    // Image-dependence control: same prompt, NO image, must differ.
    reset(&model);
    let d_noimg = describe_without_image(&model).await;
    assert_ne!(
        d1.raw_hash, d_noimg.raw_hash,
        "image turn output is not image-dependent: identical digest with and \
         without the image (vision features are not reaching generation)"
    );

    // Determinism replay.
    reset(&model);
    let (d1b, _) = run_image_turn(&model, &image).await;
    assert_eq!(d1, d1b, "image turn digest is not deterministic at T=0");
}
