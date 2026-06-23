//! Dense Qwen3.5-VL image-chat byte-identity gate for the vision
//! merge-contract genericization (S0 golden).
//!
//! The vision genericization (engine-owned VisionMerge contract, dense-first)
//! is a behavior-preserving relocation, so the dense image-chat path must emit
//! BYTE-IDENTICAL output before and after. This gate captures a stable
//! per-turn digest (raw_text hash, token count, finish reason, cached tokens)
//! for a cross-binary A/B: capture on the pre-migration commit, then re-run on
//! the migrated binary — every digest must match.
//!
//! TWO turns, because the load-bearing case is cross-turn state:
//!   * Turn 1 — `chat_session_start([image + prompt])` exercises image
//!     processing, the vision encoder, the image→token merge and M-RoPE
//!     position ids.
//!   * Turn 2 — `chat_session_continue(text)` (NO image) chains a text delta
//!     onto the image session. This is the M-RoPE cross-turn rope-delta seam:
//!     the saved `cached_rope_deltas` is read back to re-anchor positions when
//!     a text turn extends an image-bearing KV prefix (the exact state S2
//!     hoists into the engine).
//!
//! The gate makes three assertions so the fingerprint is meaningful:
//!   * IMAGE-DEPENDENCE — the same prompt with NO image must produce a
//!     DIFFERENT digest, proving the captured output is actually conditioned on
//!     the image (so any break in vision-feature injection changes the digest).
//!   * CROSS-TURN REUSE — turn 2 must report cached_tokens > 0, proving the
//!     text delta reused the image-conditioned KV prefix (the rope-delta seam).
//!   * DETERMINISM — reset + replay both turns; the digests must be
//!     byte-identical run-to-run so the printed fingerprint is a stable
//!     cross-binary gate, not a one-off sample.
//!
//! Gated on a real Qwen3.5-VL dense checkpoint + a test image. The 0.8b
//! checkpoint is vision-blind (it answers "I cannot see the image"); the 4B is
//! the smallest dense VL checkpoint that genuinely conditions on the image.
//! Run:
//!   MLX_TEST_QWEN35_VL_MODEL_PATH=.cache/models/Qwen3.5-4B-mlx \
//!   MLX_TEST_VLM_IMAGE_PATH=examples/ocr.png \
//!     cargo test -p mlx-core --test qwen3_5_vl_image_chat -- --ignored --nocapture

use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
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

fn hash8(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn oneline(s: &str) -> String {
    s.replace('\n', "\\n").chars().take(72).collect()
}

/// The cross-binary fingerprint of one chat turn: the fields that must stay
/// byte-identical across the vision refactor.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Digest {
    num_tokens: u32,
    cached_tokens: u32,
    finish_reason: String,
    raw_hash: u64,
}

/// One image+text turn, then a text-only delta on the SAME session. The delta
/// turn is the load-bearing M-RoPE cross-turn rope-delta seam. Returns each
/// turn's digest plus its raw text (for the printed fingerprint).
async fn run_two_turns(m: &Qwen3_5Model, image: &[u8]) -> (Digest, String, Digest, String) {
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

/// The SAME describe prompt with NO image, on a fresh session. Its digest must
/// differ from the image turn — the image-dependence control.
async fn describe_without_image(m: &Qwen3_5Model) -> Digest {
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
#[ignore = "needs MLX_TEST_QWEN35_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH for a Qwen3.5-VL dense checkpoint + test image"]
async fn qwen3_5_vl_image_chat_t0_capture() {
    let Ok(model_path) = std::env::var("MLX_TEST_QWEN35_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_QWEN35_VL_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_QWEN35_VL_MODEL_PATH does not exist: {model_path}"
    );
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5-VL model");

    // Full cache purge so every replay is a cold turn-1 (no cross-run reuse).
    let reset = |m: &Qwen3_5Model| {
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

    // Turn 1 must produce real output.
    assert!(d1.num_tokens > 0, "turn 1 produced zero tokens: {d1:?}");
    assert!(
        d1.finish_reason == "stop" || d1.finish_reason == "length",
        "turn 1 unexpected finish_reason: {}",
        d1.finish_reason
    );
    // Turn 2 (text delta on the image session) must produce output and reuse
    // the image-conditioned KV prefix — cached_tokens > 0 confirms the
    // cross-turn seam (image KV reuse + rope-delta re-anchor) actually ran.
    assert!(d2.num_tokens > 0, "turn 2 produced zero tokens: {d2:?}");
    assert!(
        d2.cached_tokens > 0,
        "turn 2 expected an image-KV prefix reuse but cached_tokens=0 \
         (the cross-turn rope-delta seam did not run): {d2:?}"
    );

    // --- Image-dependence control. Same prompt, NO image, fresh session: the
    // digest MUST differ, proving turn 1 is conditioned on the image (so a
    // broken vision-feature merge would change the captured fingerprint). ---
    reset(&model);
    let d_noimg = describe_without_image(&model).await;
    assert_ne!(
        d1.raw_hash, d_noimg.raw_hash,
        "turn 1 output is not image-dependent: identical digest with and \
         without the image (vision features are not reaching generation)"
    );

    // --- Pass 2: determinism. Reset + replay; digests must be byte-identical
    // run-to-run so the printed fingerprint is a stable cross-binary gate. ---
    reset(&model);
    let (d1b, _, d2b, _) = run_two_turns(&model, &image).await;
    assert_eq!(d1, d1b, "turn 1 digest is not deterministic at T=0");
    assert_eq!(d2, d2b, "turn 2 digest is not deterministic at T=0");
}
