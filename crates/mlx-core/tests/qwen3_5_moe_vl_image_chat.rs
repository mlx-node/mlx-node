//! MoE Qwen3.5-VL image-chat byte-identity gate for the vision merge-contract
//! genericization (S0 golden).
//!
//! `qwen3_5_moe` is the SECOND family that shares the qwen3.5 vision contract
//! (same image processor / vision encoder / M-RoPE + cross-turn rope-delta as
//! the dense path) — it is the real dedup target of the merge-contract refactor
//! (S3). This gate captures a stable per-turn digest so the MoE image-chat path
//! emits BYTE-IDENTICAL output before and after the relocation.
//!
//! Mirrors the dense gate exactly (see `qwen3_5_vl_image_chat.rs`): two turns
//! (image describe → text delta on the same session), an image-dependence
//! control (same prompt, no image, must differ), a cross-turn KV-reuse check,
//! and a determinism replay.
//!
//! Gated on a real Qwen3.5-VL MoE checkpoint + a test image. Run:
//!   MLX_TEST_QWEN35MOE_VL_MODEL_PATH=.cache/models/Qwen3.6-35b-a3b-UD-Q2_K_XL-mlx \
//!   MLX_TEST_VLM_IMAGE_PATH=examples/ocr.png \
//!     cargo test -p mlx-core --test qwen3_5_moe_vl_image_chat -- --ignored --nocapture

use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;
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

async fn run_two_turns(m: &Qwen3_5MoeModel, image: &[u8]) -> (Digest, String, Digest, String) {
    let t1 = m
        .chat_session_start(
            vec![user_msg("Describe this image briefly.", Some(image))],
            Some(cfg(48)),
        )
        .await
        .expect("turn 1 image chat_session_start failed");
    let d1 = Digest {
        num_tokens: t1.num_tokens,
        cached_tokens: t1.cached_tokens,
        finish_reason: t1.finish_reason.clone(),
        raw_hash: hash8(&t1.raw_text),
    };

    let t2 = m
        .chat_session_continue(
            "Answer in one word: is there text?".to_string(),
            None,
            None,
            Some(cfg(48)),
        )
        .await
        .expect("turn 2 text chat_session_continue failed");
    let d2 = Digest {
        num_tokens: t2.num_tokens,
        cached_tokens: t2.cached_tokens,
        finish_reason: t2.finish_reason.clone(),
        raw_hash: hash8(&t2.raw_text),
    };
    (d1, t1.raw_text, d2, t2.raw_text)
}

async fn describe_without_image(m: &Qwen3_5MoeModel) -> Digest {
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
#[ignore = "needs MLX_TEST_QWEN35MOE_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH for a Qwen3.5-VL MoE checkpoint + test image"]
async fn qwen3_5_moe_vl_image_chat_t0_capture() {
    let Ok(model_path) = std::env::var("MLX_TEST_QWEN35MOE_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_QWEN35MOE_VL_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_QWEN35MOE_VL_MODEL_PATH does not exist: {model_path}"
    );
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    let model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5-VL MoE model");

    let reset = |m: &Qwen3_5MoeModel| {
        tokio::task::block_in_place(|| m.reset_caches()).expect("reset_caches failed");
    };

    // --- Pass 1: capture the digest. ---
    reset(&model);
    let (d1, raw1, d2, raw2) = run_two_turns(&model, &image).await;

    println!(
        "DIGEST turn1 ntok={} cached={} finish={} sha={:016x} :: {}",
        d1.num_tokens,
        d1.cached_tokens,
        d1.finish_reason,
        d1.raw_hash,
        oneline(&raw1)
    );
    println!(
        "DIGEST turn2 ntok={} cached={} finish={} sha={:016x} :: {}",
        d2.num_tokens,
        d2.cached_tokens,
        d2.finish_reason,
        d2.raw_hash,
        oneline(&raw2)
    );

    assert!(d1.num_tokens > 0, "turn 1 produced zero tokens: {d1:?}");
    assert!(
        d1.finish_reason == "stop" || d1.finish_reason == "length",
        "turn 1 unexpected finish_reason: {}",
        d1.finish_reason
    );
    assert!(d2.num_tokens > 0, "turn 2 produced zero tokens: {d2:?}");
    assert!(
        d2.cached_tokens > 0,
        "turn 2 expected an image-KV prefix reuse but cached_tokens=0 \
         (the cross-turn rope-delta seam did not run): {d2:?}"
    );

    // Image-dependence control: same prompt, NO image, must differ.
    reset(&model);
    let d_noimg = describe_without_image(&model).await;
    assert_ne!(
        d1.raw_hash, d_noimg.raw_hash,
        "turn 1 output is not image-dependent: identical digest with and \
         without the image (vision features are not reaching generation)"
    );

    // Determinism replay.
    reset(&model);
    let (d1b, _, d2b, _) = run_two_turns(&model, &image).await;
    assert_eq!(d1, d1b, "turn 1 digest is not deterministic at T=0");
    assert_eq!(d2, d2b, "turn 2 digest is not deterministic at T=0");
}
