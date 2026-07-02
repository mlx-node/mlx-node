//! Shared harness for the synthetic MoE MTP integration tests
//! (`qwen3_5_moe_mtp_synthetic_v1.rs` / `qwen3_5_moe_mtp_synthetic_v2.rs`).
//!
//! Those are two SEPARATE test binaries on purpose: the MoE stepper reads
//! `MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY` exactly once per process (via
//! `OnceLock`), so each binary pins one flag state — v1 leaves the flag
//! unset (cycle-history default), v2 sets it before any model work
//! (committed-history).
//!
//! The harness builds a TINY random-init MoE checkpoint that ships a real
//! MTP head, reloads it, and decodes the same prompt with MTP off (plain AR
//! reference) and on. Every local real MoE-MTP checkpoint is a VLM export
//! whose config forces the block-paged KV backend — where the eager MoE MTP
//! stepper is unreachable — so this synthetic checkpoint is the only
//! always-on end-to-end route through `MoeMtpStepper`.
//!
//! What each test asserts — all four are deterministic on random weights:
//! 1. `has_mtp_weights()` is true after reload (gates the random-save
//!    `mtp.*` emission and the `mtp_num_hidden_layers` config round-trip).
//! 2. The plain AR baseline decodes exactly `max_new_tokens` tokens (EOS is
//!    unreachable by construction — see the prompt comment below).
//! 3. The MTP decode completes crash-free across every draft/verify (and,
//!    under v2, trim/commit) cycle with the same full token budget,
//!    `mtp_cycles > 1`, and a populated `mtp_mean_accepted_tokens`.
//! 4. Within-mode determinism: a second MTP decode of the same prompt on the
//!    same loaded model is byte-identical to the first. This catches
//!    allocator-dependent garbage (e.g. an out-of-bounds gather); the repeat
//!    run executes identical kernel shapes, so it must be deterministic.
//!
//! MTP==AR byte-identity is deliberately NOT asserted here. Measured on this
//! host: on random weights `mtp.text == ar.text` flips on ~10-15% of fresh
//! checkpoint draws — greedy argmax near-ties (random logits over a 250k
//! vocab have tiny top-2 gaps, and MTP's batched verify kernels round
//! differently from AR's single-token decode kernels, well above 1 ULP).
//! Exoneration of the v2 port: the flag-OFF v1 path (byte-for-byte the
//! pre-existing cycle-history code) fails at the SAME rate as v2, depth-1
//! and depth-4 MTP outputs are byte-identical to each other, and the flips
//! are late single tokens after dozens of byte-clean cycles — so the assert
//! measures kernel rounding, not port correctness (verify re-derives every
//! emitted token from the MAIN model at T=0, so the drafter's cache policy
//! cannot change which tokens are emitted). The MTP==AR byte-identity gate
//! lives in the real-weights deep test
//! (`qwen3_5_moe_mtp_committed_history.rs`), where bf16 real-distribution
//! logits keep top-2 gaps far above kernel rounding.

use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5_moe::Qwen3_5MoeConfig;
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use mlx_core::models::qwen3_5_moe::persistence::create_random_qwen35_moe_checkpoint_sync;
use mlx_core::tokenizer::ChatMessage;

/// Tiny MoE config with one MTP layer. `full_attention_interval: 1` makes
/// the single decoder layer (and the MTP layer pinned at
/// `fa_idx = interval - 1 = 0`) full-attention, so no GDN state is involved;
/// `decoder_sparse_step: 1` makes layer 0 MoE-flavored, which is the flavor
/// the MTP loader gate expects for this config. `use_block_paged_cache` is
/// left unset so the reloaded model has no paged adapter and the eager MTP
/// arm is reachable.
///
/// `vocab_size` MUST cover every id the real Qwen tokenizer can emit
/// (chat-template special tokens reach id ~248076 on the Qwen3.5 tokenizer):
/// the prompt always contains `<|im_start|>`/`<|im_end|>`, and an id past the
/// embedding table turns the prompt embedding `take()` into an out-of-bounds
/// GPU gather — undefined values that shift with allocator state, which
/// manifested as nondeterministic output for BOTH plain AR and MTP decodes
/// on an earlier `vocab_size: 1000` draft of this config.
fn tiny_mtp_config() -> Qwen3_5MoeConfig {
    Qwen3_5MoeConfig {
        vocab_size: 250_000,
        hidden_size: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        head_dim: 16,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 512,
        pad_token_id: 0,
        eos_token_id: 1,
        bos_token_id: 0,
        linear_num_value_heads: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 1,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        shared_expert_intermediate_size: Some(64),
        moe_intermediate_size: Some(64),
        norm_topk_prob: true,
        mlp_only_layers: None,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        use_block_paged_cache: None,
        n_mtp_layers: 1,
    }
}

/// Locate a real Qwen tokenizer directory under the repo's `.cache/models`.
/// The random checkpoint save writes no tokenizer, but the session chat path
/// requires one with an `<|im_end|>` special token. Mirrors the candidate
/// list in `__test__/test-model-utils.ts::findTokenizerPath`.
fn find_tokenizer_dir() -> Option<PathBuf> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    [
        ".cache/models/qwen3.5-0.8b-mlx-bf16",
        ".cache/models/qwen3.5-0.8b",
        ".cache/models/qwen3-0.6b-mlx-bf16",
        ".cache/models/qwen3-0.6b",
    ]
    .iter()
    .map(|c| repo_root.join(c))
    .find(|d| d.join("tokenizer.json").exists())
}

/// Best-effort removal of the temp checkpoint dir, including on panic
/// (assert failures unwind through the test body).
struct DirCleanup(PathBuf);

impl Drop for DirCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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

/// Build the tiny random MoE+MTP checkpoint, load it, and run the four
/// deterministic assertions documented in the module header: the MTP head
/// engages after reload, the AR baseline and the MTP decode both complete
/// the full token budget, the MTP run reports cycles/acceptance metrics,
/// and a repeat MTP decode is byte-identical to the first. `mode_label`
/// only labels messages (v1 vs v2 — the flag itself is process state owned
/// by the calling test binary).
///
/// Graceful skips (eprintln + return) ONLY for environment gaps: non-macOS
/// (no Metal) or no local Qwen tokenizer. Everything else is a hard failure —
/// in particular `has_mtp_weights() == false` after reload means the
/// random-save `mtp.*` emission or the `mtp_num_hidden_layers` config
/// round-trip regressed.
pub async fn run_synthetic_mtp_gate(mode_label: &str) {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: synthetic MoE MTP test requires Metal (macOS)");
        return;
    }
    let Some(tokenizer_dir) = find_tokenizer_dir() else {
        eprintln!(
            "skipping: no Qwen tokenizer.json found under .cache/models (download one first, \
             e.g. `yarn mlx download model -m Qwen/Qwen3-0.6B -o .cache/models/qwen3-0.6b`)"
        );
        return;
    };

    let ckpt_dir = std::env::temp_dir().join(format!(
        "mlx-moe-mtp-synth-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos()
    ));
    std::fs::create_dir_all(&ckpt_dir).expect("failed to create temp checkpoint dir");
    let _cleanup = DirCleanup(ckpt_dir.clone());
    let ckpt_path = ckpt_dir
        .to_str()
        .expect("temp checkpoint path is not valid UTF-8")
        .to_string();

    create_random_qwen35_moe_checkpoint_sync(tiny_mtp_config(), &ckpt_path)
        .expect("failed to create random MoE+MTP checkpoint");
    // The chat template lives in tokenizer_config.json next to
    // tokenizer.json; copy both (plus a standalone template if the source
    // checkpoint ships one).
    for file in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        let src = tokenizer_dir.join(file);
        if src.exists() {
            std::fs::copy(&src, ckpt_dir.join(file))
                .unwrap_or_else(|e| panic!("failed to copy {file}: {e}"));
        }
    }
    println!("synthetic MoE+MTP checkpoint created at {ckpt_path}");

    let mtp_model = Qwen3_5MoeModel::load(ckpt_path.clone())
        .await
        .expect("failed to load synthetic checkpoint (MTP run)");
    assert!(
        mtp_model.has_mtp_weights(),
        "synthetic checkpoint did not engage the MTP head after reload \
         (has_mtp_weights() == false) — the random-save mtp.* emission \
         (`create_random_qwen35_moe_checkpoint_sync`) or the \
         `mtp_num_hidden_layers` config round-trip (`save_model_sync`) regressed"
    );
    let ar_model = Qwen3_5MoeModel::load(ckpt_path)
        .await
        .expect("failed to load synthetic checkpoint (AR reference)");

    // Random weights make the CONTENT meaningless; only crash-freedom and
    // determinism matter. The vocab covers the tokenizer's <|im_end|> id (a
    // smaller vocab would make the prompt's special tokens out-of-bounds
    // embedding gathers — UB), but greedy argmax over ~250k random logits
    // virtually never lands on the one EOS id, so both runs decode exactly
    // `max_new_tokens` tokens — enough for several depth-4 MTP cycles.
    const PROMPT_TEXT: &str = "Write one short sentence about the weather.";
    const MAX_NEW_TOKENS: i32 = 32;

    // Assert 2: the plain AR baseline completes the full token budget.
    let ar_result = ar_model
        .chat_session_start(
            vec![user_message(PROMPT_TEXT)],
            Some(chat_config(MAX_NEW_TOKENS, false)),
        )
        .await
        .expect("AR reference chat_session_start failed");
    assert_eq!(
        ar_result.num_tokens, MAX_NEW_TOKENS as u32,
        "AR baseline stopped early ({mode_label}): finish_reason={} — EOS should \
         be unreachable on random logits over a 250k vocab",
        ar_result.finish_reason
    );

    // Assert 3: the MTP decode completes crash-free with the same budget and
    // reports cycle/acceptance metrics (proving the eager MTP arm engaged
    // and, under v2, that commit_mtp's forward + begin_cycle's trim ran every
    // cycle without a shape/length failure).
    let mtp_result = mtp_model
        .chat_session_start(
            vec![user_message(PROMPT_TEXT)],
            Some(chat_config(MAX_NEW_TOKENS, true)),
        )
        .await
        .expect("MTP chat_session_start failed");
    assert_eq!(
        mtp_result.num_tokens, MAX_NEW_TOKENS as u32,
        "MTP decode stopped early ({mode_label}): finish_reason={}",
        mtp_result.finish_reason
    );

    let perf = mtp_result
        .performance
        .as_ref()
        .expect("MTP performance metrics missing (reportPerformance: true)");
    assert!(
        perf.mtp_mean_accepted_tokens.is_some(),
        "has_mtp_weights() was true but no MTP cycle ran \
         (mtp_mean_accepted_tokens is None) — the eager MTP arm did not engage \
         ({mode_label})"
    );
    let cycles = perf.mtp_cycles.unwrap_or(0);
    assert!(
        cycles > 1,
        "expected multiple MTP cycles, got {cycles} ({mode_label})"
    );

    // Assert 4: within-mode determinism. A same-prompt `chat_session_start`
    // on the same model takes the engine's exact-match-as-miss route (the
    // zero-delta guard in `engine/session.rs`: full cache reset + full
    // re-prefill), so this repeat run executes the identical kernel shapes
    // as the first — cold-equivalent, hence bitwise-deterministic. Any
    // divergence here is real (allocator-dependent UB such as an
    // out-of-bounds gather), never an argmax near-tie.
    let mtp_repeat = mtp_model
        .chat_session_start(
            vec![user_message(PROMPT_TEXT)],
            Some(chat_config(MAX_NEW_TOKENS, true)),
        )
        .await
        .expect("repeat MTP chat_session_start failed");
    assert_eq!(
        mtp_repeat.text, mtp_result.text,
        "repeat MTP decode diverged from the first MTP run ({mode_label}) — \
         identical kernel shapes must be deterministic; suspect \
         allocator-dependent UB (e.g. an out-of-bounds gather)"
    );
    assert_eq!(
        mtp_repeat.num_tokens, mtp_result.num_tokens,
        "repeat MTP decode token count diverged ({mode_label})"
    );
    let repeat_cycles = mtp_repeat
        .performance
        .as_ref()
        .and_then(|p| p.mtp_cycles)
        .unwrap_or(0);

    println!(
        "synthetic moe mtp ({mode_label}): cycles={} mean_accepted_tokens={:?} \
         num_tokens={} finish_reason={} repeat_cycles={repeat_cycles} \
         repeat_byte_identical=true ar_num_tokens={}",
        cycles,
        perf.mtp_mean_accepted_tokens,
        mtp_result.num_tokens,
        mtp_result.finish_reason,
        ar_result.num_tokens
    );
}
