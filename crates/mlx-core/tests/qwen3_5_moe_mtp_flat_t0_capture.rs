//! Qwen3.5 MoE flat-MTP byte-identity gate for the MTP-engine genericization.
//!
//! S5 relocates the MoE eager-MTP propose/verify loop out of the
//! `moe_eager_mtp_decode!` wrapper macro into the engine-owned `run_mtp_turn`
//! (driven by a `MoeMtpStepper`), exactly as S2b–S4 did for the dense family.
//! The migration is behavior-preserving, so the MoE flat-MTP path must emit
//! BYTE-IDENTICAL output before and after. This gate captures a stable
//! per-(prompt,depth) digest (raw_text hash, token count, finish reason, mean
//! accepted) for a cross-binary A/B: capture at the pre-migration commit, then
//! re-run on the migrated binary — every digest must match.
//!
//! NOTE: this does NOT assert MTP == AR. On bf16/quantized checkpoints the MTP
//! verify path (windowed-fp32) can flip a greedy near-tie relative to per-step
//! AR even in the original code, so MTP-vs-AR is not a byte-identity invariant
//! here; MTP-vs-MTP-across-the-refactor is.
//!
//! Text-only prompts drive the plain flat MTP sites (`chat_tokens_delta_sync`
//! / `chat_stream_tokens_delta_sync_inner`, gated on `paged_adapter.is_none()`),
//! NOT the vision-MTP cores (those need an image input). All four MoE MTP sites
//! expand the SAME wrapper macro, so a byte-identical plain-flat A/B proves the
//! shared `MoeMtpStepper` is correct; the vision cores differ only in prefill.
//!
//! An observed `mtp_mean_accepted_tokens` confirms the flat MTP path actually
//! executed (the gate is not vacuous).
//!
//! Run (MoE checkpoint WITH an MTP head):
//!   MLX_TEST_MOE_MODEL_PATH=/Volumes/P4510/models/qwen3.6-35b-a3b-mxfp8-mtp \
//!     cargo test -p mlx-core --test qwen3_5_moe_mtp_flat_t0_capture -- --ignored --nocapture

use std::path::Path;

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use mlx_core::tokenizer::ChatMessage;

fn cfg(max_new_tokens: i32, enable_mtp: bool, depth: i32) -> ChatConfig {
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
        // The session API requires reuse_cache; independence between runs is
        // enforced by an explicit reset_caches() before each generation.
        reuse_cache: Some(true),
        enable_mtp: Some(enable_mtp),
        mtp_depth: if enable_mtp { Some(depth) } else { None },
        mtp_adaptive_depth: Some(false),
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

fn hash8(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn oneline(s: &str) -> String {
    s.replace('\n', "\\n").chars().take(72).collect()
}

const PROMPTS: &[&str] = &[
    "Count from 1 to 30, space separated.",
    "Write a short paragraph about the ocean.",
    "List five fruits, one per line.",
];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MODEL_PATH = a Qwen3.5 MoE checkpoint WITH an MTP head"]
async fn moe_flat_mtp_matches_ar_t0() {
    let Ok(model_path) = std::env::var("MLX_TEST_MOE_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MOE_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MOE_MODEL_PATH does not exist: {model_path}"
    );

    let model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 MoE model");
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint has no MTP head (has_mtp_weights() == false)");
        return;
    }

    // Independence between generations: purge all cache state so every run is
    // a cold turn-1 (no cross-run prefix reuse), keeping the digest stable.
    let reset = |m: &Qwen3_5MoeModel| {
        tokio::task::block_in_place(|| m.reset_caches()).expect("reset_caches failed");
    };

    let mut any_mtp_ran = false;
    for (pi, prompt) in PROMPTS.iter().enumerate() {
        for depth in [1i32, 2, 3] {
            reset(&model);
            let mtp = model
                .chat_session_start(vec![user_message(prompt)], Some(cfg(64, true, depth)))
                .await
                .expect("MTP chat_session_start failed");

            let acc = mtp
                .performance
                .as_ref()
                .and_then(|p| p.mtp_mean_accepted_tokens);
            let mtp_ran = acc.is_some();
            any_mtp_ran |= mtp_ran;
            let tps = mtp
                .performance
                .as_ref()
                .map(|p| p.decode_tokens_per_second)
                .unwrap_or(0.0);

            // Stable per-(prompt,depth) MTP fingerprint for the cross-binary
            // A/B. sha/ntok/finish/acc must be identical pre- and post-migration.
            println!(
                "DIGEST p{pi} d{depth} ntok={} finish={} acc={:.4} mtp_ran={} tps={:.1} sha={:016x} :: {}",
                mtp.num_tokens,
                mtp.finish_reason,
                acc.unwrap_or(0.0),
                mtp_ran,
                tps,
                hash8(&mtp.raw_text),
                oneline(&mtp.raw_text),
            );
        }
    }

    assert!(
        any_mtp_ran,
        "MTP never ran on any prompt (mtp_mean_accepted_tokens always absent) — \
         gate would be vacuous; check the checkpoint has a usable MTP head and the \
         flat path is active"
    );
}

/// STREAM == SYNC gate for the MoE flat eager-MTP path.
///
/// S5 relocated the wrapper macro's streaming arm into the engine-owned
/// `run_mtp_turn` so the SYNC and STREAMING MoE flat-MTP sites now share ONE
/// loop with a sink switch. This proves the streaming site emits the exact
/// same text + token count as the sync site for every (prompt, depth): the
/// streamed concatenation of all non-done chunks must equal the sync
/// `raw_text`, and the streamed token count must equal the sync `num_tokens`.
///
/// Since the sync path is already pinned byte-identical by
/// `moe_flat_mtp_matches_ar_t0` (the cross-binary digest A/B), proving
/// streaming == sync transitively proves streaming is byte-identical too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MODEL_PATH = a Qwen3.5 MoE checkpoint WITH an MTP head"]
async fn moe_flat_mtp_stream_matches_sync() {
    let Ok(model_path) = std::env::var("MLX_TEST_MOE_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MOE_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MOE_MODEL_PATH does not exist: {model_path}"
    );

    let model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 MoE model");
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint has no MTP head (has_mtp_weights() == false)");
        return;
    }

    let reset = |m: &Qwen3_5MoeModel| {
        tokio::task::block_in_place(|| m.reset_caches()).expect("reset_caches failed");
    };

    for (pi, prompt) in PROMPTS.iter().enumerate() {
        for depth in [1i32, 2, 3] {
            // --- SYNC MTP: the already-baseline-identical path. ---
            reset(&model);
            let sync = model
                .chat_session_start(vec![user_message(prompt)], Some(cfg(64, true, depth)))
                .await
                .expect("MTP chat_session_start failed");
            let sync_acc = sync
                .performance
                .as_ref()
                .and_then(|p| p.mtp_mean_accepted_tokens);
            assert!(
                sync_acc.is_some(),
                "p{pi} d{depth}: sync MTP did not run (mtp_mean_accepted_tokens absent) — \
                 the flat MTP path is not active, gate would be vacuous"
            );

            // --- STREAMING MTP: the relocated streaming sink. ---
            reset(&model);
            let (_handle, mut rx) = model
                .chat_stream_session_start_for_test(
                    vec![user_message(prompt)],
                    Some(cfg(64, true, depth)),
                )
                .expect("MTP chat_stream_session_start_for_test dispatch failed");

            let mut stream_concat = String::new();
            let mut stream_num_tokens: Option<u32> = None;
            let mut stream_done_text: Option<String> = None;
            let mut stream_done_raw: Option<String> = None;
            let mut saw_done = false;
            while let Some(result) = rx.recv().await {
                let chunk = result.expect("stream chunk error");
                if chunk.done {
                    stream_num_tokens = chunk.num_tokens;
                    stream_done_text = Some(chunk.text.clone());
                    stream_done_raw = chunk.raw_text.clone();
                    saw_done = true;
                    break;
                }
                stream_concat.push_str(&chunk.text);
            }
            assert!(
                saw_done,
                "p{pi} d{depth}: streaming run never reached done=true"
            );
            let stream_num_tokens = stream_num_tokens.expect("done chunk must carry num_tokens");
            let stream_done_text = stream_done_text.expect("done chunk must carry text");

            println!(
                "STREAM==SYNC p{pi} d{depth} sync_ntok={} stream_ntok={} sync_acc={:.4} \
                 sync_raw_sha={:016x} stream_concat_sha={:016x} sync_text_sha={:016x} \
                 stream_text_sha={:016x} :: {}",
                sync.num_tokens,
                stream_num_tokens,
                sync_acc.unwrap_or(0.0),
                hash8(&sync.raw_text),
                hash8(&stream_concat),
                hash8(&sync.text),
                hash8(&stream_done_text),
                oneline(&stream_concat),
            );

            assert_eq!(
                stream_num_tokens, sync.num_tokens,
                "p{pi} d{depth}: streamed num_tokens ({stream_num_tokens}) != sync num_tokens ({}) \
                 — streaming and sync diverged on token count",
                sync.num_tokens,
            );

            assert_eq!(
                stream_concat, sync.raw_text,
                "p{pi} d{depth}: streamed concatenation != sync raw_text — streaming and sync \
                 diverged on the emitted token stream"
            );

            assert_eq!(
                stream_done_text, sync.text,
                "p{pi} d{depth}: streaming done-chunk text != sync text — the assembled \
                 clean_text diverged"
            );
            if let Some(stream_done_raw) = stream_done_raw {
                assert_eq!(
                    stream_done_raw, sync.raw_text,
                    "p{pi} d{depth}: streaming done-chunk raw_text != sync raw_text"
                );
            }
        }
    }
}
