//! Dense-flat MTP byte-identity gate for the MTP-engine genericization.
//!
//! The MTP-engine migration is a behavior-preserving relocation, so the dense
//! flat-MTP path must emit BYTE-IDENTICAL output before and after. This gate
//! captures a stable per-(prompt,depth) digest of the MTP output (raw_text
//! hash, token count, finish reason, mean accepted) for a cross-binary A/B:
//! capture at the pre-migration commit, then re-run on the migrated binary —
//! every digest must match.
//!
//! NOTE: this does NOT assert MTP == AR. On bf16/nvfp4 checkpoints the MTP
//! verify path (windowed-fp32) can flip a greedy near-tie relative to per-step
//! AR even in the original code, so MTP-vs-AR is not a byte-identity invariant
//! here; MTP-vs-MTP-across-the-refactor is.
//!
//! MTP only runs when `paged_adapter.is_none()`, so an observed
//! `mtp_mean_accepted_tokens` confirms the FLAT path actually executed (the
//! gate is not vacuous).
//!
//! Run (dense checkpoint WITH an MTP head):
//!   MLX_TEST_MODEL_PATH=/Volumes/P4510/models/qwen3.5-4b-nvfp4-mtp \
//!     cargo test -p mlx-core --test qwen3_5_mtp_flat_t0_capture -- --ignored --nocapture

use std::path::Path;

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
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
#[ignore = "needs MLX_TEST_MODEL_PATH = a Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn dense_flat_mtp_matches_ar_t0() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MODEL_PATH does not exist: {model_path}"
    );

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint has no MTP head (has_mtp_weights() == false)");
        return;
    }

    // Independence between generations: purge all cache state so every run is
    // a cold turn-1 (no cross-run prefix reuse), keeping the AR-vs-MTP
    // comparison deterministic and clean.
    let reset = |m: &Qwen3_5Model| {
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

/// STREAM == SYNC gate for the dense flat eager-MTP path.
///
/// S3 relocated the macro's streaming arm into the engine-owned
/// `run_mtp_turn` so the SYNC and STREAMING dense-flat MTP sites now share
/// ONE loop with a sink switch. This proves the streaming site emits the
/// exact same text + token count as the sync site for every
/// (prompt, depth): the streamed concatenation of all non-done chunks must
/// equal the sync `result.text`, and the streamed token count (the
/// done-chunk's `num_tokens`) must equal the sync `result.num_tokens`.
///
/// Since the sync path is already pinned byte-identical by
/// `dense_flat_mtp_matches_ar_t0` (the cross-binary digest A/B), proving
/// streaming == sync transitively proves streaming is byte-identical too.
///
/// Both runs `reset_caches()` (via `block_in_place`) first so each is a
/// cold turn-1 with no cross-run prefix reuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH = a Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn dense_flat_mtp_stream_matches_sync() {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MODEL_PATH does not exist: {model_path}"
    );

    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint has no MTP head (has_mtp_weights() == false)");
        return;
    }

    // Independence between runs: purge all cache state so every generation is
    // a cold turn-1 (no cross-run prefix reuse).
    let reset = |m: &Qwen3_5Model| {
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
                    // The terminal chunk carries num_tokens + the assembled
                    // clean_text (`text`) and `raw_text`; it is NOT part of the
                    // per-token concat.
                    stream_num_tokens = chunk.num_tokens;
                    stream_done_text = Some(chunk.text.clone());
                    stream_done_raw = chunk.raw_text.clone();
                    saw_done = true;
                    break;
                }
                // Concatenate EVERY non-done delta (per-token streamed text +
                // the post-loop residual flush). With include_reasoning=true the
                // suppression gate is a no-op, so these deltas reconstruct the
                // full RAW decoded text — i.e. `raw_text`, NOT the
                // reasoning-stripped `clean_text` the sync `text` field holds.
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

            // Token count parity: streamed token count == sync token count.
            assert_eq!(
                stream_num_tokens, sync.num_tokens,
                "p{pi} d{depth}: streamed num_tokens ({stream_num_tokens}) != sync num_tokens ({}) \
                 — streaming and sync diverged on token count",
                sync.num_tokens,
            );

            // Emitted-text parity: the streamed per-token concatenation (incl.
            // the residual flush) reconstructs the FULL raw decoded text, so it
            // must equal the sync `raw_text` byte-for-byte — the direct proof
            // that streaming and sync committed the identical token stream
            // (`run_mtp_turn`'s shared loop emitting through the streaming sink
            // vs the sync no-op sink).
            assert_eq!(
                stream_concat, sync.raw_text,
                "p{pi} d{depth}: streamed concatenation != sync raw_text — streaming and sync \
                 diverged on the emitted token stream"
            );

            // Assembled-text parity: the streaming done-chunk's clean_text
            // (`text`) must equal the sync `text`, and its `raw_text` the sync
            // `raw_text` — the finalize/reasoning-split tail is identical too.
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
