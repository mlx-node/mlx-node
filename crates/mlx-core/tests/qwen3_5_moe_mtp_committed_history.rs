//! Gated integration test for Qwen3.5-MoE's MTP committed-history v2
//! within-turn persistence (the `MoeMtpStepper::use_committed` /
//! `commit_mtp` / `begin_cycle` port — see the `MoeMtpStepper` struct doc in
//! `crates/mlx-core/src/models/qwen3_5_moe/model.rs`).
//!
//! v2 is opt-in behind `MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY` (default off).
//! This test does NOT force the flag — it inherits it from the process env, so
//! the SAME test binary runs as v1 (flag unset) or v2 (flag=1) depending on how
//! it is launched. That makes it the same-binary A/B correctness harness:
//!   - flag unset → v1 (fresh drafter cache each cycle)
//!   - flag=1     → v2 (persistent drafter cache across cycles)
//!
//! Both modes MUST produce output byte-identical to a plain AR decode.
//!
//! Speculative decoding's accept/reject verification always re-derives ground
//! truth from the MAIN model's own forward pass, so the emitted token SEQUENCE
//! at T=0 is invariant to the drafter's cache-retention policy: a broken
//! `commit_mtp` / `begin_cycle` port can only crash (a drafter-cache
//! shape/length mismatch) or silently degrade the drafter's OWN proposals
//! (worse accept rate) — never change which tokens get emitted. This test
//! therefore gates crash/shape/corruption, NOT accept-rate; the perf win is
//! validated separately by a flag-off/flag-on same-checkpoint A/B (deferred to
//! the consolidated cooled-GPU bench).
//!
//! It requires a checkpoint that actually ENGAGES the MoE MTP drafter — i.e.
//! `has_mtp_weights()` is true (config declares `mtp_num_hidden_layers`/
//! `num_nextn_predict_layers` > 0 AND the `mtp.*` head tensors load, including
//! `.scales` siblings for any packed head) AND the turn runs the eager MTP
//! arm. If the checkpoint at `MLX_TEST_MOE_MTP_MODEL_PATH` does not engage
//! MTP, the test SKIPS (mirrors the dense reference `qwen3_5_delta_chat.rs`)
//! rather than asserting on a silently-AR run. In particular, VLM MoE-MTP
//! checkpoints force the block-paged backend, where the eager MTP stepper is
//! unreachable — the head loads but never engages, so the test skips there.
//!
//! This is the opt-in DEEP gate for a future text MoE-MTP checkpoint; the
//! always-on gates for the eager MoE MTP path (both flag states) are the
//! synthetic random-checkpoint pair `qwen3_5_moe_mtp_synthetic_v1.rs` /
//! `qwen3_5_moe_mtp_synthetic_v2.rs`.
//!
//! Run it manually with:
//!
//! ```shell
//! # v2 (committed-history on):
//! MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY=1 \
//! MLX_TEST_MOE_MTP_MODEL_PATH=/absolute/path/to/moe-mtp-checkpoint \
//!     cargo test -p mlx-core --test qwen3_5_moe_mtp_committed_history \
//!     -- --ignored --nocapture --test-threads=1
//! # v1 (baseline): same command without MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY.
//! ```
//!
//! Without `MLX_TEST_MOE_MTP_MODEL_PATH` the test early-returns and passes
//! trivially so it still compiles as part of `cargo test`.

use std::path::Path;

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use mlx_core::tokenizer::ChatMessage;

fn chat_config(max_new_tokens: i32, enable_mtp: bool) -> ChatConfig {
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
        thinking_token_budget: Some(0),
        include_reasoning: Some(false),
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: Some(enable_mtp),
        mtp_depth: Some(4),
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
        audio: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MTP_MODEL_PATH pointing to a real Qwen3.5-MoE-A3B MTP checkpoint"]
async fn mtp_matches_ar_after_multi_cycle_decode() {
    let Ok(model_path) = std::env::var("MLX_TEST_MOE_MTP_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MOE_MTP_MODEL_PATH unset (point it at an ABSOLUTE path to a \
             Qwen3.5-MoE-A3B checkpoint that ships an MTP head, e.g. \
             /abs/path/to/qwen3.6-35b-a3b-mxfp8-mtp)"
        );
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MOE_MTP_MODEL_PATH does not exist: {model_path}"
    );
    println!("MLX_TEST_MOE_MTP_MODEL_PATH resolved to: {model_path}");

    // v2 is read from the env (default off) by the model itself — we do NOT set
    // it here, so the same binary is the v1/v2 A/B harness. Report which mode
    // this run exercises.
    let v2_on = std::env::var("MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    println!(
        "committed-history mode: {}",
        if v2_on {
            "v2 (flag on)"
        } else {
            "v1 (flag off / default)"
        }
    );

    const PROMPT_TEXT: &str = "Write a short numbered list (6 items) of steps to plan a \
         weekend hiking trip. Keep each step to one sentence.";

    // Load the MTP model first and check it actually engages the MoE MTP
    // drafter. If not (config declares no mtp layers, or the mtp.* head — incl.
    // its .scales for a packed head — did not load), SKIP rather than assert on
    // a silently-AR decode. Mirrors the dense reference guard in
    // `qwen3_5_delta_chat.rs`.
    let mtp_model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 MoE model (MTP)");
    println!(
        "MTP model loaded; has_mtp_weights() = {}",
        mtp_model.has_mtp_weights()
    );
    if !mtp_model.has_mtp_weights() {
        eprintln!(
            "skipping MTP assertions: checkpoint at {model_path} does not engage the MoE MTP \
             drafter (has_mtp_weights() == false — its config declares no mtp layer count, or the \
             mtp.* head tensors / their .scales siblings did not load). Point \
             MLX_TEST_MOE_MTP_MODEL_PATH at a checkpoint whose MTP head loads."
        );
        return;
    }

    // Reference: plain autoregressive decode (MTP disabled).
    let ar_model = Qwen3_5MoeModel::load(model_path)
        .await
        .expect("failed to load Qwen3.5 MoE model (AR reference)");
    println!("AR reference model loaded");
    let ar_result = ar_model
        .chat_session_start(
            vec![user_message(PROMPT_TEXT)],
            Some(chat_config(200, false)),
        )
        .await
        .expect("AR reference chat_session_start failed");

    // MTP-enabled decode over the SAME prompt/params — 200 tokens at depth 4
    // runs many `begin_cycle` / `commit_mtp` cycles, exactly the path this
    // fix changes.
    let mtp_result = mtp_model
        .chat_session_start(
            vec![user_message(PROMPT_TEXT)],
            Some(chat_config(200, true)),
        )
        .await
        .expect("MTP chat_session_start failed");

    let perf = mtp_result
        .performance
        .as_ref()
        .expect("MTP performance metrics missing (reportPerformance: true)");

    // MTP actually ran (not a silent AR fallback): the acceptance summary is
    // only populated when at least one MTP cycle was recorded. This is a
    // graceful skip, not a failure: every local MoE-MTP checkpoint is a VLM
    // export whose config forces the block-paged KV backend, and the paged
    // MoE core has no MTP stepper — the eager MTP arm this test targets is
    // unreachable there, so the head loads but never engages.
    if perf.mtp_mean_accepted_tokens.is_none() {
        eprintln!(
            "skipping MTP assertions: the MTP head loaded but no MTP cycle ran on this turn \
             (mtp_mean_accepted_tokens is None). This is expected for VLM MoE-MTP checkpoints \
             (vision_config forces the block-paged backend, where the eager MoE MTP stepper is \
             unreachable). Point MLX_TEST_MOE_MTP_MODEL_PATH at a TEXT MoE-MTP checkpoint (no \
             vision_config, use_block_paged_cache unset/false) to exercise this deep gate; the \
             always-on gate is the synthetic pair qwen3_5_moe_mtp_synthetic_v1/_v2."
        );
        return;
    }

    assert_eq!(
        mtp_result.text, ar_result.text,
        "MTP-enabled decode diverged from the AR reference at T=0 — the \
         committed-history port likely corrupted `mtp_caches` (check \
         `MoeMtpStepper::commit_mtp` / `begin_cycle` trim targets)"
    );
    assert_eq!(mtp_result.num_tokens, ar_result.num_tokens);
    let cycles = perf.mtp_cycles.unwrap_or(0);
    assert!(cycles > 1, "expected multiple MTP cycles, got {cycles}");
    println!(
        "moe mtp ({}): cycles={} mean_accepted_tokens_total={:?} mean_depth={:?}",
        if v2_on { "v2" } else { "v1" },
        cycles,
        perf.mtp_mean_accepted_tokens_total,
        perf.mtp_mean_depth
    );
}
