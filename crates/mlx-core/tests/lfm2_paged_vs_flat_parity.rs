//! Real-weights paged-vs-flat numerical-equivalence gate for LFM2.
//!
//! LFM2 is the second of the two models with full forward dispatch wired
//! through `forward_paged_adapter`; Qwen3 is the other (see
//! `qwen3_paged_vs_flat_parity.rs`). LFM2's hybrid architecture — 10 conv
//! layers + 6 full_attention layers — means only the attention layers
//! route through the adapter; conv layers stay on `Lfm2LayerCache::Conv`.
//! Greedy parity over real weights is the gate that proves the per-layer
//! routing logic agrees with the flat path on the attention layers
//! without disturbing conv state on the others.
//!
//! The test mirrors `qwen3_paged_vs_flat_parity.rs` test cases (a) and
//! (c). The standalone logit-max-abs-diff variant (test b in the spec) is
//! intentionally skipped for LFM2 — its hybrid arch makes per-layer logit
//! comparison harder, and greedy parity already catches the same class of
//! regressions. The Qwen3 file's docstring covers the spec deviation.
//!
//! Gated on `MLX_TEST_MODEL_PATH` so a plain `cargo test --ignored`
//! without the env var still passes (the early-return short-circuits the
//! body before any model load).
//!
//! Run locally with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/lfm2.5-1.2b-thinking-mlx \
//!     cargo test -p mlx-core --test lfm2_paged_vs_flat_parity \
//!     -- --ignored --nocapture
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::models::lfm2::model::Lfm2Model;
use mlx_core::models::qwen3_5::model::ChatConfig;
use mlx_core::tokenizer::ChatMessage;

// ---------------------------------------------------------------------------
// Test fixture helpers (mirrors qwen3_paged_vs_flat_parity.rs)
// ---------------------------------------------------------------------------

/// Copy the source LFM2 checkpoint directory into a fresh tempdir under
/// the workspace `target/` and optionally patch `config.json` to enable
/// the block-paged adapter.
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

    let dst = workspace_target.join(format!("paged-parity-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    // Symlink large weight files instead of copying them; only config.json
    // is mutated per-clone. Avoids OOM on disk for multi-GB checkpoints.
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

    // Always write `use_block_paged_cache` explicitly (in BOTH branches),
    // mirroring `lfm2_compiled_e2e.rs`. Without this the flat clone
    // (`use_block_paged == false`) would leave the bf16 source config — which
    // OMITS the key — untouched, and `Lfm2Inner::new`'s `unwrap_or(true)`
    // would silently load the PAGED path. That made the parity tests compare
    // paged-vs-paged and miss flat-path regressions. Pinning the flag forces
    // the flat clone onto the genuine flat path (`Some(false)`).
    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path)
        .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
    let mut cfg: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(use_block_paged);
    if use_block_paged {
        // 256 MB pool + 16-token blocks: enough to hold the test's tiny
        // prompts × all attention layers in LFM2-1.2B (head_dim=128,
        // kv_heads varies by layer kind). The adapter only allocates
        // capacity for `full_attention` layers, so the budget is tighter
        // than Qwen3's at the same MB.
        cfg["paged_cache_memory_mb"] = serde_json::Value::from(256u32);
        cfg["paged_block_size"] = serde_json::Value::from(16u32);
    }
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty)
        .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;

    Ok(dst)
}

fn parity_chat_config(max_new_tokens: i32) -> ChatConfig {
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

fn parity_prompts() -> [&'static str; 4] {
    [
        "Say hi in one short word.",
        "What is 2 + 3? Answer with just the number.",
        "Name a primary color.",
        "Complete: the sky is",
    ]
}

fn resolve_source_model() -> Option<PathBuf> {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/lfm2.5-1.2b-thinking-mlx)"
        );
        return None;
    };
    let p = PathBuf::from(&model_path);
    if !p.exists() {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH does not exist: {}",
            p.display()
        );
        return None;
    }
    Some(p)
}

// ---------------------------------------------------------------------------
// Test (a): greedy-decode token parity over 4 prompts
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_paged_vs_flat_greedy_token_parity() {
    let Some(src) = resolve_source_model() else {
        return;
    };

    let flat_dir = match clone_model_dir(&src, "lfm2-flat", false) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for flat path: {e}"),
    };
    let paged_dir = match clone_model_dir(&src, "lfm2-paged", true) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for paged path: {e}"),
    };

    let flat_model = Lfm2Model::load_from_dir(&flat_dir.to_string_lossy())
        .await
        .expect("failed to load flat-path LFM2 model");
    let paged_model = Lfm2Model::load_from_dir(&paged_dir.to_string_lossy())
        .await
        .expect("failed to load paged-path LFM2 model");

    let prompts = parity_prompts();
    for (idx, prompt) in prompts.iter().enumerate() {
        // Each prompt is fresh; no explicit `reset_caches()` here —
        // see the Qwen3 parity test for the rationale (sync NAPI calls
        // panic inside a tokio runtime, and `chat_session_start`'s
        // verify_cache_prefix handles the implicit reset on a miss).

        let cfg_flat = parity_chat_config(32);
        let cfg_paged = parity_chat_config(32);

        let r_flat = flat_model
            .chat_session_start(vec![user_message(prompt)], Some(cfg_flat))
            .await
            .unwrap_or_else(|e| panic!("flat chat_session_start failed (prompt #{idx}): {e:?}"));
        let r_paged = paged_model
            .chat_session_start(vec![user_message(prompt)], Some(cfg_paged))
            .await
            .unwrap_or_else(|e| panic!("paged chat_session_start failed (prompt #{idx}): {e:?}"));

        eprintln!(
            "prompt #{idx} ({prompt:?}): flat num_tokens={} paged num_tokens={} | \
             flat finish={} paged finish={}",
            r_flat.num_tokens, r_paged.num_tokens, r_flat.finish_reason, r_paged.finish_reason
        );

        if r_flat.text != r_paged.text {
            let first_diff = r_flat
                .text
                .as_bytes()
                .iter()
                .zip(r_paged.text.as_bytes().iter())
                .position(|(a, b)| a != b);
            panic!(
                "TEXT MISMATCH on prompt #{idx} ({prompt:?}). \
                 first_diff_byte={first_diff:?}\n\
                 FLAT  ({} tokens) text={:?}\n\
                 PAGED ({} tokens) text={:?}",
                r_flat.num_tokens, r_flat.text, r_paged.num_tokens, r_paged.text,
            );
        }
        assert_eq!(
            r_flat.num_tokens, r_paged.num_tokens,
            "num_tokens mismatch on prompt #{idx} ({prompt:?}): flat={} paged={}",
            r_flat.num_tokens, r_paged.num_tokens,
        );
        assert_eq!(
            r_flat.finish_reason, r_paged.finish_reason,
            "finish_reason mismatch on prompt #{idx}: flat={} paged={}",
            r_flat.finish_reason, r_paged.finish_reason,
        );
    }

    eprintln!("LFM2 greedy parity: all {} prompts matched", prompts.len());
}

// ---------------------------------------------------------------------------
// Test (c): two-turn dialog parity (exercises prefix-reuse on attention
// layers + correct conv-state preservation across both paths)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_paged_vs_flat_prefix_reuse_parity() {
    let Some(src) = resolve_source_model() else {
        return;
    };

    let flat_dir = match clone_model_dir(&src, "lfm2-flat-2t", false) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for flat path: {e}"),
    };
    let paged_dir = match clone_model_dir(&src, "lfm2-paged-2t", true) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for paged path: {e}"),
    };

    let flat_model = Lfm2Model::load_from_dir(&flat_dir.to_string_lossy())
        .await
        .expect("failed to load flat-path LFM2 model");
    let paged_model = Lfm2Model::load_from_dir(&paged_dir.to_string_lossy())
        .await
        .expect("failed to load paged-path LFM2 model");

    let prompt1 = "Say hi in one short word.";
    let r1_flat = flat_model
        .chat_session_start(vec![user_message(prompt1)], Some(parity_chat_config(32)))
        .await
        .expect("turn 1 flat chat_session_start failed");
    let r1_paged = paged_model
        .chat_session_start(vec![user_message(prompt1)], Some(parity_chat_config(32)))
        .await
        .expect("turn 1 paged chat_session_start failed");

    assert_eq!(
        r1_flat.text, r1_paged.text,
        "turn 1 text mismatch: flat={:?} paged={:?}",
        r1_flat.text, r1_paged.text
    );
    assert_eq!(
        r1_flat.num_tokens, r1_paged.num_tokens,
        "turn 1 num_tokens mismatch: flat={} paged={}",
        r1_flat.num_tokens, r1_paged.num_tokens
    );

    let user2 = "And in another word?";
    let r2_flat = flat_model
        .chat_session_continue(user2.to_string(), None, Some(parity_chat_config(32)))
        .await
        .expect("turn 2 flat chat_session_continue failed");
    let r2_paged = paged_model
        .chat_session_continue(user2.to_string(), None, Some(parity_chat_config(32)))
        .await
        .expect("turn 2 paged chat_session_continue failed");

    eprintln!(
        "two-turn parity: turn2 flat num_tokens={} cached={} | paged num_tokens={} cached={}",
        r2_flat.num_tokens, r2_flat.cached_tokens, r2_paged.num_tokens, r2_paged.cached_tokens,
    );

    if r2_flat.text != r2_paged.text {
        panic!(
            "TURN-2 TEXT MISMATCH (prefix-reuse divergence between paths)\n\
             FLAT  ({} tokens, cached={}) text={:?}\n\
             PAGED ({} tokens, cached={}) text={:?}",
            r2_flat.num_tokens,
            r2_flat.cached_tokens,
            r2_flat.text,
            r2_paged.num_tokens,
            r2_paged.cached_tokens,
            r2_paged.text,
        );
    }
    assert_eq!(
        r2_flat.num_tokens, r2_paged.num_tokens,
        "turn 2 num_tokens mismatch (prefix-reuse divergence): flat={} paged={}",
        r2_flat.num_tokens, r2_paged.num_tokens,
    );
}

// ---------------------------------------------------------------------------
// Telemetry regression: LFM2 paged prefillTokensPerSecond must be full-prompt
// scale, not attention-suffix scale, on a warm cross-request prefix-cache hit.
//
// The paged prefill reprocesses the FULL prompt through the conv layers every
// turn (run_paged_prefill_chunk Pass 1), so ttft measures full-prompt work.
// The pre-fix code divided that ttft into the attention SUFFIX
// (tokens.len()-cached_prefix_len), under-reporting prefill tok/s by the
// cache-hit ratio (~37 vs ~thousands). This guards the fix that uses the
// full-prompt count (tokens.len()) as the numerator.
// ---------------------------------------------------------------------------

/// Like `parity_chat_config` but with `report_performance: true` so the
/// returned `ChatResult.performance` is populated.
fn perf_chat_config(max_new_tokens: i32) -> ChatConfig {
    let mut cfg = parity_chat_config(max_new_tokens);
    cfg.report_performance = Some(true);
    cfg
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_paged_prefill_tps_is_full_prompt_scale_on_warm_reuse() {
    let Some(src) = resolve_source_model() else {
        return;
    };

    // Paged adapter ON (the path with the telemetry bug).
    let paged_dir = match clone_model_dir(&src, "lfm2-paged-perf", true) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone model dir for paged path: {e}"),
    };
    let paged_model = Lfm2Model::load_from_dir(&paged_dir.to_string_lossy())
        .await
        .expect("failed to load paged-path LFM2 model");

    // A reasonably long turn-1 prompt so the turn-2 cached prefix dwarfs the
    // new suffix — that is the regime where suffix-vs-full-prompt diverge.
    let prompt1 = "Write a short paragraph about the history of the printing press, \
                   covering Gutenberg, movable type, and the spread of literacy across \
                   Europe in the fifteenth and sixteenth centuries.";
    let _r1 = paged_model
        .chat_session_start(vec![user_message(prompt1)], Some(perf_chat_config(32)))
        .await
        .expect("turn 1 paged chat_session_start failed");

    // Turn 2: a FRESH paged session re-submitting the IDENTICAL prompt. On a
    // paged model the BlockAllocator reuses the content-addressed prefix, so
    // `cached_tokens` covers ~the whole prompt while the new attention suffix is
    // tiny — yet paged prefill still reprocesses the FULL prompt through the conv
    // layers (run_paged_prefill_chunk Pass 1), so ttft stays full-prompt scale.
    // This exercises the `chat_sync_core_paged` START path that the telemetry fix
    // touches, NOT the `chat_session_continue` delta path (which legitimately
    // forwards only the delta via `chat_tokens_delta_sync` and is left unchanged).
    let r2 = paged_model
        .chat_session_start(vec![user_message(prompt1)], Some(perf_chat_config(32)))
        .await
        .expect("turn 2 paged chat_session_start failed");

    let perf = r2
        .performance
        .expect("performance must be present when report_performance:true");

    eprintln!(
        "paged perf regression: prompt_tokens={} cached_tokens={} ttft_ms={:.2} \
         prefill_tps={:.2}",
        r2.prompt_tokens, r2.cached_tokens, perf.ttft_ms, perf.prefill_tokens_per_second,
    );

    // (b) The cache-hit context is still reported (warm reuse actually happened).
    assert!(
        r2.cached_tokens > 0,
        "expected a warm prefix-cache hit on turn 2 (cached_tokens>0), got {}",
        r2.cached_tokens,
    );

    // (a) prefill tok/s must be full-prompt scale. Reconstruct the full-prompt
    // and suffix expectations from the SAME ttft the engine measured, then
    // assert the reported value tracks the full prompt, not the suffix.
    //
    // ttft_ms can be ~0 only on a degenerate empty prefill; guard so the test
    // fails loudly rather than dividing by zero.
    assert!(
        perf.ttft_ms > 0.0,
        "ttft_ms must be positive for the throughput check, got {}",
        perf.ttft_ms,
    );
    let ttft_s = perf.ttft_ms / 1000.0;
    let full_prompt_tps = r2.prompt_tokens as f64 / ttft_s;
    let suffix_len = (r2.prompt_tokens as i64 - r2.cached_tokens as i64).max(0) as f64;
    let suffix_tps = suffix_len / ttft_s;

    // The reported value must be at least half the full-prompt rate. A
    // suffix-scale numerator (which is >5x smaller here, since cached_tokens
    // dominates) would fall far below this bar and fail.
    assert!(
        perf.prefill_tokens_per_second >= 0.5 * full_prompt_tps,
        "prefill_tokens_per_second ({:.2}) must be full-prompt scale (>= 0.5 * {:.2}); \
         a suffix-scale numerator would report ~{:.2}",
        perf.prefill_tokens_per_second,
        full_prompt_tps,
        suffix_tps,
    );
}

// ---------------------------------------------------------------------------
// Regression: a quantized LFM2 checkpoint loaded with NO config override must
// default to the FLAT decode path.
//
// The default is resolved in `Lfm2Inner::load_from_dir` from the authoritative
// `.scales` tensor signal (NOT config metadata), the same signal that gates
// compiled registration. This proves the wiring end-to-end on real weights and
// guards against a quantized checkpoint silently landing on the slow eager-PAGED
// path (the bug this branch removes). Because detection keys on tensors, a
// checkpoint whose `quantization` block lacks top-level `bits`/`mode` (per-layer
// only) is handled identically — the metadata shape is never consulted.
//
// Gated on its OWN env var so it only runs when the operator explicitly points
// at a QUANTIZED checkpoint (a bf16 checkpoint would correctly stay paged and
// fail this assertion).
// ---------------------------------------------------------------------------
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_QUANTIZED_MODEL_PATH pointing to a real QUANTIZED LFM2 checkpoint"]
async fn lfm2_quantized_default_load_takes_flat_path() {
    let Ok(model_path) = std::env::var("MLX_TEST_QUANTIZED_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_QUANTIZED_MODEL_PATH unset (point it at a quantized LFM2 \
             checkpoint, e.g. an mxfp8/affine-4bit convert output)"
        );
        return;
    };
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: MLX_TEST_QUANTIZED_MODEL_PATH does not exist: {model_path}");
        return;
    }

    // DEFAULT load — no config override, no clone. The on-disk config.json has
    // `use_block_paged_cache` unset; the loader must flip it to flat because the
    // weights carry `.scales`.
    let model = Lfm2Model::load_from_dir(&model_path)
        .await
        .expect("failed to load quantized LFM2 model");

    assert!(
        !model.has_block_paged_cache(),
        "a quantized LFM2 checkpoint with use_block_paged_cache unset must default to FLAT \
         (has_block_paged_cache()==false); got paged — the .scales-keyed default did not fire"
    );
}
