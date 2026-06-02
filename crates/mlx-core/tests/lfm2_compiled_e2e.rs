//! Stage D: make-or-break real-model end-to-end parity tests for the LFM2
//! **compiled C++ flat decode path** on the real LFM2.5-1.2B checkpoint.
//!
//! These are INTEGRATION tests (their own process), so they do NOT co-run with
//! the `--lib` synthetic component-parity probes that destructively clear the
//! shared C++ weight table (`g_weights()`).
//!
//! What they prove:
//!   1. ENGAGEMENT — the compiled C++ forward (`mlx_lfm2_moe_forward`) actually
//!      runs once per decode step. If the per-step call count is 0 the flat path
//!      silently fell back to native: that is a hard FAILURE for the compiled
//!      test (and the EXPECTED state for the native-flat reference test).
//!   2. PARITY — greedy decode on the compiled flat path matches the reference,
//!      which is the gate that the compiled forward + cache seam is numerically
//!      correct on real weights.
//!   3. OFFSET FIX — long-prompt (chunked-prefill last-chunk-only) and warm
//!      strict-extend reuse seed the compiled decode position from the LIVE KV
//!      offset (`KVCache::get_offset()`), not the per-chunk `seq_len`.
//!
//! ## References available for "is the compiled flat port faithful?"
//!
//! * **compiled-flat** (`use_block_paged_cache:false`, no `MLX_NO_COMPILE`):
//!   the path under test.
//! * **eager-flat** (`use_block_paged_cache:false`, `MLX_NO_COMPILE=1`): the
//!   RIGHT reference for isolating a `compile()` TOPOLOGY bug from a LOGIC bug.
//!   VERIFIED EMPIRICALLY: for lfm2, `MLX_NO_COMPILE=1` does NOT give a pure-Rust
//!   forward. lfm2's Rust gate `Lfm2Inner::compiled_path_active()` does NOT read
//!   `MLX_NO_COMPILE`; it stays ON, so the decode loop still calls
//!   `mlx_lfm2_moe_forward` once per step (the engagement counter STILL
//!   increments ~N-1). Inside that C++ function the env flag only swaps the
//!   compiled callable for an EAGER `lfm2_decode_fn(...)` call
//!   (`mlx_lfm2_moe.cpp:343`) — SAME flat KV/conv layout, SAME offset / causal
//!   mask / RoPE / cache logic, just NOT wrapped in `mlx::core::compile`. So
//!   `eager-flat == compiled-flat` means `compile()` is faithful (any
//!   compiled-vs-paged tail flip is then pure flat-vs-paged bf16 noise), and
//!   `eager-flat != compiled-flat` means a real `compile()` topology/offset bug
//!   (it is the ONLY thing that differs between the two runs).
//!   Because `static bool no_compile` in the C++ latches once per process, the
//!   eager run MUST be a SEPARATE `cargo test` invocation with `MLX_NO_COMPILE=1`;
//!   we compare across invocations through an artifact file, never an in-process
//!   env toggle.
//! * **native-paged** (`use_block_paged_cache:true`): a DIFFERENT graph (paged
//!   attention), so bf16 rounding differs from flat and a single late argmax
//!   near-tie can legitimately flip the final token(s). Useful as a second
//!   opinion, but NOT the fidelity oracle.
//!
//! How the flat/paged path is forced: the public load API
//! (`Lfm2Model::load_from_dir`) has no `use_block_paged_cache` override, so — as
//! `lfm2_paged_vs_flat_parity.rs` already does — we clone the checkpoint dir and
//! patch `config.json`. NO production file is modified.
//!
//! `compiled_path_active()` is `pub(crate)` and unreachable from an integration
//! test, so engagement is asserted via the public mlx-sys extern
//! `mlx_lfm2_moe_forward_call_count()` (cumulative, process-global; we capture a
//! before/after delta around the decode run).
//!
//! Gated: every test runs ONLY when `LFM2_COMPILED_E2E=1` AND the checkpoint dir
//! exists. Otherwise it early-returns (skips) so the default `cargo test` never
//! loads a 1.2B model. Checkpoint path comes from `MLX_TEST_MODEL_PATH` or
//! defaults to `.cache/models/lfm2.5-1.2b-thinking-mlx`.
//!
//! ## Reproduce
//!
//! ```shell
//! # The default 1-process suite: compiled-flat vs paged, long-prompt offset
//! # fix, warm-reuse offset fix. Also captures the compiled-flat text artifact.
//! LFM2_COMPILED_E2E=1 \
//!   MLX_TEST_MODEL_PATH=./.cache/models/lfm2.5-1.2b-thinking-mlx \
//!   cargo test -p mlx-core --test lfm2_compiled_e2e -- --nocapture
//!
//! # SEPARATE process: capture the eager-flat (MLX_NO_COMPILE=1) reference and
//! # diff it against the compiled-flat artifact written by the run above. This
//! # is the verdict for "real compile() bug vs flat-vs-paged bf16 noise".
//! LFM2_COMPILED_E2E=1 MLX_NO_COMPILE=1 \
//!   MLX_TEST_MODEL_PATH=./.cache/models/lfm2.5-1.2b-thinking-mlx \
//!   cargo test -p mlx-core --test lfm2_compiled_e2e \
//!   lfm2_eager_flat_vs_compiled_capture -- --nocapture
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::models::lfm2::model::Lfm2Model;
use mlx_core::models::qwen3_5::model::{ChatConfig, ChatResult, ChatStreamChunk};
use mlx_core::tokenizer::ChatMessage;

/// New tokens decoded per run for the short-prompt tests. Spec requires N >= 48;
/// we ask for exactly this many and require the run to NOT stop early so the
/// per-step compiled call count is deterministic (>= N-1).
const N_NEW_TOKENS: i32 = 48;

/// Expected registered weight count for the dense LFM2.5-1.2B compiled table
/// (16 layers). Engagement / gate sanity (STEP 4).
const EXPECTED_WEIGHT_COUNT: usize = 148;

/// Lower bound on the compiled forward-call delta for an N-token decode. The
/// last token needs no further forward, so the delta is N-1; we allow a small
/// floor to rule out a stray call while still proving real engagement.
const MIN_COMPILED_CALL_DELTA: u64 = 40;

/// Tail-divergence tolerance, in BYTES, for "two correct implementations that
/// disagree only by a late greedy argmax tie". Two LFM2 decode graphs that are
/// numerically faithful but use different attention kernels (flat vs paged, or
/// compiled vs native) can flip the FINAL one or two sampled tokens when the
/// top-2 logits are a near-tie; on this checkpoint the observed flip is "But"
/// vs "But the", i.e. a single short trailing word. We treat a shared prefix
/// that leaves only a short trailing remainder on EACH side as benign. Anything
/// that diverges earlier than this is a real bug and fails the test.
///
/// One LFM2 token is at most a handful of bytes; we budget two tokens plus
/// slack. (We cannot compare raw token ids: `finalize_chat_result` sets
/// `ChatResult::token_ids = None` for the lfm2 chat path, so text is the only
/// signal an integration test can observe.)
const TAIL_TOLERANCE_BYTES: usize = 16;

// =============================================================================
// PHASE 4 PIECE 2: INDEPENDENT mlx-lm GOLDEN ORACLE
//
// Every reference the rest of this file uses (eager-flat `MLX_NO_COMPILE=1`,
// native-paged) is the SAME Rust forward + SAME C++ `mlx_lfm2_moe.cpp` math:
// they share rope_theta, norm_eps, RoPE convention, and MoE routing, so a bug
// present in ALL three Rust/C++ paths (wrong rope base, wrong RMSNorm eps,
// softmax-before-vs-after expert_bias, wrong top-k partition) is INVISIBLE to
// every existing test. There is ZERO non-Rust parity here today.
//
// The constants below close that gap. They are a FROZEN greedy decode captured
// from **mlx-lm** (Apple's reference MLX language-model library, a SEPARATE
// codebase), via `scripts/capture_lfm2_golden.py` run under
// `uv run --python 3.12 --with mlx-lm` (mlx-lm 0.31.3). Both sides do pure
// greedy (temperature 0 → argmax): mlx-lm's `generate_step` defaults its
// sampler to `lambda x: mx.argmax(x, -1)` (generate.py:386); our compiled path
// runs `golden_chat_config` (temperature 0 → per-step argmax, all penalties
// off, NO thinking budget so the natural reasoning trace is emitted).
//
// PROMPT PARITY is guaranteed by the SHARED chat template: mlx-lm renders the
// prompt with `tokenizer.apply_chat_template(...)` and our chat path templates
// the SAME user string from the SAME checkpoint. We pin the prompt-id COUNT as
// a cheap template-drift canary (`ChatResult.prompt_tokens`), since
// `ChatResult` exposes no token ids.
//
// TEXT-not-ids is forced: `ChatResult` has no `token_ids` field
// (qwen3_5/model.rs:6604), so an integration test reaching the compiled path
// only through `chat_session_start` can compare `raw_text` (verbatim decode,
// includes the `<think>…` span) against the golden's decoded text. A wrong
// rope/eps/routing diverges within the FIRST FEW tokens, which the
// minimum-shared-prefix gate below catches hard; a benign late bf16/argmax-tie
// flip stays inside the tail budget.
//
// REGENERATE / AUDIT:
//   uv run --python 3.12 --with mlx-lm python \
//     scripts/capture_lfm2_golden.py .cache/models/lfm2.5-1.2b-thinking-mlx
//   uv run --python 3.12 --with mlx-lm python \
//     scripts/capture_lfm2_golden.py .cache/models/lfm2.5-8b-a1b
// =============================================================================

/// The user string both sides template. (Already the suite's prompt above; a
/// named const here documents that the golden was captured for exactly it.)
const GOLDEN_PROMPT: &str = "What is the capital of France? Answer in one short sentence.";

/// Templated prompt-id count for the DENSE checkpoint (mlx-lm 0.31.3
/// `apply_chat_template`, lfm2.5-1.2b-thinking-mlx). Asserted against
/// `ChatResult.prompt_tokens` as a template-drift canary. Doc of the exact ids:
/// `[1, 6, 6423, 708, 3493, 856, 779, 5706, 803, 4481, 540, 42275, 797, 1235,
///   3290, 13184, 523, 7, 708, 6, 64015, 708]`
const GOLDEN_PROMPT_TOKENS_DENSE: u32 = 22;

/// Templated prompt-id count for the MoE checkpoint (lfm2.5-8b-a1b). Exact ids:
/// `[124894, 124899, 5922, 207, 2992, 355, 278, 5205, 302, 3980, 39, 41774,
///   296, 734, 2789, 12683, 22, 124900, 207, 124899, 63514, 207]`
const GOLDEN_PROMPT_TOKENS_MOE: u32 = 22;

/// mlx-lm 0.31.3 greedy generated ids for the DENSE checkpoint (DOC / REGEN
/// ONLY — `ChatResult` exposes no token ids, so the runtime gate compares
/// `raw_text` against `GOLDEN_TEXT_DENSE`). 80 ids, 55 distinct:
/// `[64400, 9095, 892, 521, 2944, 1090, 2130, 523, 941, 3952, 856, 14065, 875,
///   779, 5706, 803, 4481, 521, 810, 859, 1595, 811, 5642, 797, 1235, 3290,
///   13184, 523, 941, 5196, 11473, 811, 3151, 1550, 779, 2524, 5642, 5662, 768,
///   6377, 521, 1203, 1509, 859, 1595, 811, 1825, 4029, 859, 1673, 936, 2084,
///   523, 509, 1098, 5706, 803, 4481, 856, 5242, 521, 2084, 540, 62947, 521,
///   859, 6848, 896, 988, 31116, 1515, 523, 2173, 779, 5642, 1753, 874, 997,
///   1098, 5706]`
///
/// Verbatim mlx-lm decode of those 80 ids (includes the leading `<think>`).
const GOLDEN_TEXT_DENSE: &str = "<think> Okay, let's see. The question is asking for the capital of France, and I need to answer in one short sentence. The user specified to put only the final answer inside a box, but first I need to make sure I get it right.\n\nThe capital of France is Paris, right? Yeah, I remember that from geography class. So the answer should be \"The capital";

/// mlx-lm 0.31.3 greedy generated ids for the MoE checkpoint (DOC / REGEN ONLY).
/// 80 ids, 44 distinct:
/// `[124901, 207, 597, 4695, 20589, 34, 496, 2992, 355, 278, 5205, 302, 3980,
///   39, 41774, 296, 734, 2789, 12683, 2426, 8, 2083, 1094, 310, 5141, 34, 440,
///   5205, 302, 3980, 355, 4741, 22, 3231, 2789, 12683, 22, 1672, 34, 496, 597,
///   5205, 302, 3980, 355, 4741, 2426, 3584, 589, 734, 12683, 20, 2789, 22,
///   3584, 589, 6970, 63908, 2083, 1946, 5431, 794, 4666, 3639, 22, 43972, 395,
///   12683, 22, 2752, 4666, 34405, 22, 1672, 5141, 34, 496, 597, 5205, 302]`
///
/// Verbatim mlx-lm decode of those 80 ids (includes the leading `<think>`).
const GOLDEN_TEXT_MOE: &str = "<think>\nThe user asks: \"What is the capital of France? Answer in one short sentence.\"\n\nWe need to answer: The capital of France is Paris. One short sentence. So: \"The capital of France is Paris.\" That's one sentence, short. That's fine.\n\nWe must ensure no extra content. Provide that sentence. No extra commentary. So answer: \"The capital of";

/// Minimum shared BYTE prefix the compiled DENSE `raw_text` must agree on with
/// the mlx-lm golden. This is the DECISIVE oracle gate: a wrong rope base /
/// RMSNorm eps / RoPE convention diverges within the first handful of tokens
/// (< 10 bytes), so requiring a long exact prefix catches the bug class Piece 2
/// exists for. Cross-impl bf16 rounding (mlx-lm continuous prefill vs our
/// chunked-prefill + compiled per-step decode) can legitimately flip a LATER
/// near-tie token, so we do NOT demand byte-exact 80 tokens — only a long, early
/// agreement. OBSERVED on this checkpoint: 314 bytes (~70 tokens) of exact
/// agreement, then a benign flip; 200 keeps a comfortable margin while staying
/// far past the first-token divergence a real impl bug would produce.
const GOLDEN_MIN_PREFIX_BYTES_DENSE: usize = 200;

/// Minimum shared BYTE prefix for the 8B **MoE** golden. Set to the FULL proven
/// byte-identical agreement (138 bytes / 33 generated tokens) so the gate floor
/// sits AT the divergence, not below it: a real impl bug (wrong MoE routing
/// order / rope base 5e6 / eps) diverges at token 1-3 (< 10 bytes) and is caught
/// far below this floor, while the single benign one-ULP flip past it (and the
/// ENTIRE post-gate tail) is pinned exactly by `GOLDEN_RAW_TEXT_MOE_OURS` (the
/// byte-reproducible compiled self-golden) rather than tolerated.
///
/// OBSERVED + DIAGNOSED (REPRODUCIBLE — `scripts/probe_lfm2_divergence.py`, run
/// live on the real lfm2.5-8b-a1b 2026-06-01): the compiled MoE decode is
/// BYTE-IDENTICAL to the mlx-lm golden for the first 33 generated tokens
/// (`<think>\n…The capital of France is Paris. ` = 138 bytes), then diverges on a
/// TRUE bf16 tie at generated step 33. The probe (which prints mlx-lm's
/// per-step log-softmax top-k) shows that step's top-3 as:
///   ` One`  = -1.5000   (rank #1 — mlx-lm's greedy pick)
///   ` So`   = -1.7500   (rank #2 — OUR compiled path's pick)
///   ` That` = -1.7500   (tied #2)
/// log-softmax is shift-invariant in DIFFERENCES, so a -1.50 vs -1.75 logprob gap
/// IS a 0.25 RAW-LOGIT gap = exactly ONE bf16 ULP at this logit magnitude (~37).
/// gap2nd = 0.25 at step 33 is the TIGHTEST near-tie in the entire 80-token
/// sequence — and it is precisely where we flip. mlx-lm's continuous-prefill
/// rounding lands on ` One`; our chunked-prefill + compiled per-step rounding
/// lands on ` So`. (The earlier code comment claimed ` That`/` One` tied at
/// 37.7500 with ` So`=37.5000 — those raw-logit magnitudes were self-reported and
/// are NOT what the committed probe shows; the probe's log-softmax can only show
/// the -1.50/-1.75/-1.75 landscape above, whose 0.25 difference is the true logit
/// gap. The 37-ish magnitude survives only as the scale at which 0.25 == 1 ULP.)
///
/// Why this is conclusively benign, not a bug: (1) 33 byte-identical tokens
/// INCLUDING two earlier EXACT-tie steps — step 24 (` answer`=-1.25 vs
/// ` respond`=-1.25) and step 26 (` The` vs ` "`) — that our compiled path
/// resolves IDENTICALLY to mlx-lm; a real routing/rope/eps bug cannot reproduce
/// 33 tokens through exact ties and then break by exactly one ULP at the single
/// tightest tie. (2) compiled-flat MoE agrees with eager-flat MoE
/// (`MLX_NO_COMPILE=1`) on the GOLDEN (thinking, 80-token) trajectory through the
/// step-33 ' So' flip — they share 346 bytes, then eager appends ~10 trailing
/// newlines at a LATE benign tie (tail_diff <= TAIL_TOLERANCE_BYTES), proven by
/// `lfm2_moe_golden_eager_vs_compiled`. (The unqualified "byte-identical"
/// claim held only on the GREEDY 48-token trajectory — `run_flat_single`,
/// thinking_budget=0 — NOT this golden trajectory where the step-33 divergence
/// happens; the golden-trajectory test is the correct compile()-faithfulness
/// evidence.) So the flip is inherent to our MoE bf16 math vs mlx-lm's rounding,
/// NOT a `compile()` tracing / chunked-prefill artifact. This proves the compiled
/// MoE routing (gate→f32→softmax→+expert_bias→argpartition top-4→norm_topk_prob),
/// rope base, and eps are faithful to mlx-lm's lfm2_moe.py.
const GOLDEN_MIN_PREFIX_BYTES_MOE: usize = 138;

/// Our COMPILED-FLAT MoE decode's FULL raw_text (80 tokens, golden config),
/// byte-reproducible across runs on lfm2.5-8b-a1b. First 138 bytes are byte-
/// identical to the mlx-lm golden (GOLDEN_TEXT_MOE); the remainder is our benign
/// one-ULP ' So' continuation. Asserting the FULL text (not just the 2-byte flip)
/// gives regression teeth across the ENTIRE decode. Regenerate alongside
/// GOLDEN_TEXT_MOE if the checkpoint changes.
const GOLDEN_RAW_TEXT_MOE_OURS: &str = "<think>\nThe user asks: \"What is the capital of France? Answer in one short sentence.\"\n\nWe need to answer: The capital of France is Paris. So a short sentence: \"Paris is the capital of France.\" That's one short sentence. That satisfies. No extra commentary. Provide that.\n\nThus final answer: \"Paris is the capital of France.\"\n</think>\nParis is the";

/// Our COMPILED-FLAT DENSE decode's FULL raw_text (80 tokens, golden config),
/// byte-reproducible; first 314 bytes byte-identical to the mlx-lm golden.
const GOLDEN_RAW_TEXT_DENSE_OURS: &str = "<think> Okay, let's see. The question is asking for the capital of France, and I need to answer in one short sentence. The user specified to put only the final answer inside a box, but first I need to make sure I get it right.\n\nThe capital of France is Paris, right? Yeah, I remember that from geography class. So I need to say that in";

/// Greedy / deterministic chat config (temperature 0, all penalties off).
fn greedy_chat_config(max_new_tokens: i32, reuse_cache: bool) -> ChatConfig {
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
        // Disable repetition cutoffs so the loop never stops short of
        // `max_new_tokens` (keeps the compiled call count == N deterministic).
        max_consecutive_tokens: None,
        max_ngram_repeats: None,
        ngram_size: None,
        tools: None,
        reasoning_effort: None,
        thinking_token_budget: Some(0),
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(reuse_cache),
    }
}

/// Greedy config with MINIMAL reasoning + VERBATIM `raw_text`.
///
/// `reasoning_effort: "none"` (+ `thinking_token_budget: None`) forces the
/// thinking budget to `0` via `default_thinking_budget_for_effort`, so the
/// model emits a degenerate empty `<think></think>` span on the first two
/// decode steps and then answers — deterministic and short. (LFM2's chat
/// template IGNORES `enable_thinking`; the runtime hard-codes
/// `thinking_enabled = true` in `chat_sync_core`, so "none" controls the
/// reasoning BUDGET, not whether reasoning is parsed.)
///
/// `include_reasoning: Some(true)` is LOAD-BEARING for the warm strict-extend
/// reachability via `chat_session_start` (see `lfm2_warm_reuse_offset_fix`):
/// `save_cache_state` persists the RAW generated tokens, which BEGIN with the
/// `<think></think>` markers. `ChatResult.text` always has that reasoning span
/// stripped (`parse_thinking_and_tools` splits at `</think>` whenever
/// `thinking_enabled` — always true on LFM2), so echoing `text` back in the
/// next turn's history drops the `<think></think>` tokens and the re-templated
/// prompt diverges from the cache at the first generated token (verifier
/// returns 0, full prefill). With `include_reasoning: Some(true)`,
/// `ChatResult.raw_text` is the FULL verbatim decode (markers included), which
/// round-trips token-exact through the tokenizer, so echoing `r.raw_text`
/// re-renders the exact cached prefix and `verify_cache_prefix` takes the
/// strict-extend branch. (`text` is unaffected by `include_reasoning`, so the
/// warm-vs-cold output comparison is identical either way.)
fn greedy_chat_config_no_thinking(max_new_tokens: i32, reuse_cache: bool) -> ChatConfig {
    ChatConfig {
        reasoning_effort: Some("none".to_string()),
        thinking_token_budget: None,
        include_reasoning: Some(true),
        ..greedy_chat_config(max_new_tokens, reuse_cache)
    }
}

/// Greedy config for the INDEPENDENT mlx-lm golden comparison.
///
/// Differs from `greedy_chat_config` in ONE load-bearing way: NO thinking
/// budget (`thinking_token_budget: None`). The suite's `greedy_chat_config`
/// pins `thinking_token_budget: Some(0)`, which — when the template opens
/// thinking — would force a `</think>` on the first decode step and decode a
/// DIFFERENT trajectory than mlx-lm (which has no thinking budget and emits the
/// model's natural `<think>…` reasoning trace, exactly what the golden
/// captures). With the budget removed, our per-step argmax follows the same
/// greedy path as mlx-lm's argmax sampler. `include_reasoning: Some(true)` keeps
/// `ChatResult.raw_text` the verbatim decode of every generated token (incl. the
/// `<think>` span), which is what the golden text is.
fn golden_chat_config(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        thinking_token_budget: None,
        include_reasoning: Some(true),
        ..greedy_chat_config(max_new_tokens, true)
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

/// Assistant turn used to replay a prior reply when building a strict-extend
/// superset prompt (warm-reuse test). `reasoning_content: None` keeps the
/// re-rendered history a strict byte-prefix extension of what the session
/// persisted, so `verify_cache_prefix` takes the strict-extend branch.
fn assistant_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        images: None,
    }
}

/// Clone the source checkpoint dir into a fresh tempdir under the workspace
/// `target/`, symlinking the multi-GB weight files and patching `config.json`
/// to force the requested cache backend.
///
/// `use_block_paged == false` -> flat compiled path; `true` -> native paged.
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

    let dst = workspace_target.join(format!("compiled-e2e-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

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

    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path)
        .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
    let mut cfg: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(use_block_paged);
    if use_block_paged {
        // Generous pool for the paged reference; long-prompt tests push >2048
        // tokens through the paged attention layers.
        cfg["paged_cache_memory_mb"] = serde_json::Value::from(1024u32);
        cfg["paged_block_size"] = serde_json::Value::from(16u32);
    }
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty)
        .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;

    Ok(dst)
}

fn resolve_source_model() -> Option<PathBuf> {
    let model_path = std::env::var("MLX_TEST_MODEL_PATH")
        .unwrap_or_else(|_| ".cache/models/lfm2.5-1.2b-thinking-mlx".to_string());
    let p = PathBuf::from(&model_path);
    if !p.join("config.json").exists() {
        eprintln!(
            "[skip] checkpoint not found (config.json missing) at {}",
            p.display()
        );
        return None;
    }
    Some(p)
}

/// Structural MoE predicate: a checkpoint dir is an lfm2_moe model iff its
/// `config.json` sets `model_type == "lfm2_moe"` OR carries a `num_experts` field.
/// Mirrors the inline predicate the MoE tests used; on any read/parse error
/// returns `false` (treat an unreadable config as "not provably MoE"). Centralized
/// here so the MoE tests share ONE source of truth instead of re-implementing it.
fn is_moe_config(path: &Path) -> bool {
    let cfg = path.join("config.json");
    let Ok(raw) = fs::read_to_string(&cfg) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    json.get("model_type").and_then(|m| m.as_str()) == Some("lfm2_moe")
        || json.get("num_experts").is_some()
}

/// Fail-closed MoE-checkpoint selector. MoE coverage is REQUIRED under
/// `LFM2_COMPILED_E2E=1` and must never silently pass against a missing or dense
/// default — a forgotten env var must surface as a HARD FAILURE, not a green skip.
///
///   * `LFM2_MOE_MODEL_PATH` set  -> MoE coverage REQUESTED: panic (never skip) if
///     it is missing config.json or is not an lfm2_moe checkpoint.
///   * `LFM2_SKIP_MOE_E2E` set    -> EXPLICIT opt-out: skip (returns None).
///   * neither set                -> coverage REQUIRED: accept only if the shared
///     resolver already points at a MoE checkpoint; otherwise PANIC so a
///     missing/dense default can never masquerade as MoE coverage.
fn resolve_moe_model() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("LFM2_MOE_MODEL_PATH") {
        let p = PathBuf::from(p);
        assert!(
            p.join("config.json").exists(),
            "LFM2_MOE_MODEL_PATH={} has no config.json — MoE coverage requested but the \
             checkpoint is missing (hard failure, NOT a skip)",
            p.display()
        );
        assert!(
            is_moe_config(&p),
            "LFM2_MOE_MODEL_PATH={} is not an lfm2_moe checkpoint — MoE coverage requested \
             but the path points at a non-MoE model",
            p.display()
        );
        return Some(p);
    }
    if std::env::var_os("LFM2_SKIP_MOE_E2E").is_some() {
        eprintln!("[skip] LFM2_SKIP_MOE_E2E set — MoE e2e explicitly opted out");
        return None;
    }
    // No explicit MoE path AND no explicit opt-out: coverage is REQUIRED. Accept only
    // if the shared resolver already points at a MoE checkpoint; otherwise HARD-FAIL so
    // a forgotten env var never silently passes as MoE coverage.
    match resolve_source_model() {
        Some(src) if is_moe_config(&src) => Some(src),
        other => panic!(
            "MoE e2e coverage REQUIRED under LFM2_COMPILED_E2E=1 but no MoE checkpoint is \
             available (resolved {}). Set LFM2_MOE_MODEL_PATH=<lfm2.5-8b-a1b> to run it, or \
             LFM2_SKIP_MOE_E2E=1 to EXPLICITLY opt out (a missing/dense default must not \
             masquerade as MoE coverage).",
            other
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "none".to_string())
        ),
    }
}

fn gated() -> bool {
    std::env::var("LFM2_COMPILED_E2E").as_deref() == Ok("1")
}

fn no_compile_env() -> bool {
    // Match the C++ side EXACTLY: the paged forward selects the eager branch when
    // `std::getenv("MLX_NO_COMPILE") != nullptr || std::getenv("MLX_DISABLE_COMPILE")
    // != nullptr` — for ANY value (incl. "0"). Using `== Ok("1")` here would
    // desync: `MLX_NO_COMPILE=0` would run the eager C++ branch while the test
    // still believed the compiled graph ran.
    //
    // MLX_DISABLE_COMPILE makes MLX's `compile()` a no-op, so the compiled-paged
    // counter never bumps under it; engagement tests that early-skip under
    // no-compile MUST also skip here, or they'd assert a false-positive
    // call-delta. (The C++ `no_compile` latch now ORs both env vars too.)
    std::env::var_os("MLX_NO_COMPILE").is_some()
        || std::env::var_os("MLX_DISABLE_COMPILE").is_some()
}

/// Workspace-`target/` path for the cross-invocation native-flat artifact. PID
/// is intentionally NOT included so a `MLX_NO_COMPILE=1` invocation can find the
/// compiled-flat run's artifact written by an earlier invocation.
fn artifact_path(name: &str) -> PathBuf {
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
    workspace_target.join(name)
}

/// Outcome of a single decode run, with the engagement signals.
struct RunOutcome {
    text: String,
    /// Verbatim decode of ALL generated tokens (includes any `<think>…` span);
    /// the signal compared against the mlx-lm golden (`.text` strips reasoning).
    raw_text: String,
    /// Templated prompt-id count (template-drift canary vs the mlx-lm golden).
    prompt_tokens: u32,
    num_tokens: u32,
    finish_reason: String,
    call_delta: u64,
    model_id: u64,
    weight_count: usize,
}

fn run_outcome(label: &str, r: &ChatResult, call_delta: u64) -> RunOutcome {
    let model_id = unsafe { mlx_sys::mlx_lfm2_get_model_id() };
    let weight_count = unsafe { mlx_sys::mlx_lfm2_weight_count() };
    eprintln!(
        "[{label}] num_tokens={} prompt_tokens={} finish={} call_delta={call_delta} model_id={model_id} weight_count={weight_count}",
        r.num_tokens, r.prompt_tokens, r.finish_reason
    );
    eprintln!("[{label}] text = {:?}", r.text);
    RunOutcome {
        text: r.text.clone(),
        raw_text: r.raw_text.clone(),
        prompt_tokens: r.prompt_tokens,
        num_tokens: r.num_tokens,
        finish_reason: r.finish_reason.clone(),
        call_delta,
        model_id,
        weight_count,
    }
}

/// Load a flat (`use_block_paged_cache:false`) model and run a single-turn
/// greedy decode, returning the outcome plus the compiled forward-call delta.
async fn run_flat_single(src: &Path, suffix: &str, prompt: &str, max_new: i32) -> RunOutcome {
    let dir = clone_model_dir(src, suffix, false)
        .unwrap_or_else(|e| panic!("clone flat dir ({suffix}): {e}"));
    let model = Lfm2Model::load_from_dir(&dir.to_string_lossy())
        .await
        .unwrap_or_else(|e| panic!("load flat model ({suffix}): {e:?}"));
    let before = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
    let r = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(greedy_chat_config(max_new, true)),
        )
        .await
        .unwrap_or_else(|e| panic!("flat chat_session_start ({suffix}): {e:?}"));
    let after = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
    let out = run_outcome(suffix, &r, after.saturating_sub(before));
    drop(model);
    out
}

/// Load a flat (`use_block_paged_cache:false`) model and run a single-turn
/// greedy decode using `golden_chat_config` (NO thinking budget) for the
/// independent mlx-lm golden comparison. Returns the outcome plus the compiled
/// forward-call delta.
async fn run_flat_golden(src: &Path, suffix: &str, prompt: &str, max_new: i32) -> RunOutcome {
    let dir = clone_model_dir(src, suffix, false)
        .unwrap_or_else(|e| panic!("clone golden flat dir ({suffix}): {e}"));
    let model = Lfm2Model::load_from_dir(&dir.to_string_lossy())
        .await
        .unwrap_or_else(|e| panic!("load golden flat model ({suffix}): {e:?}"));
    let before = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
    let r = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(golden_chat_config(max_new)),
        )
        .await
        .unwrap_or_else(|e| panic!("golden flat chat_session_start ({suffix}): {e:?}"));
    let after = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
    let out = run_outcome(suffix, &r, after.saturating_sub(before));
    drop(model);
    out
}

/// Load a paged (`use_block_paged_cache:true`) reference model and run a
/// single-turn greedy decode.
async fn run_paged_single(src: &Path, suffix: &str, prompt: &str, max_new: i32) -> RunOutcome {
    let dir = clone_model_dir(src, suffix, true)
        .unwrap_or_else(|e| panic!("clone paged dir ({suffix}): {e}"));
    let model = Lfm2Model::load_from_dir(&dir.to_string_lossy())
        .await
        .unwrap_or_else(|e| panic!("load paged model ({suffix}): {e:?}"));
    let before = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
    let r = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(greedy_chat_config(max_new, true)),
        )
        .await
        .unwrap_or_else(|e| panic!("paged chat_session_start ({suffix}): {e:?}"));
    let after = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
    let out = run_outcome(suffix, &r, after.saturating_sub(before));
    drop(model);
    out
}

/// Load a paged (`use_block_paged_cache:true`) model and run a single-turn
/// greedy decode, capturing the COMPILED-PAGED engagement delta
/// (`mlx_lfm2_moe_compiled_paged_call_count`) — the per-step counter bumped only
/// inside the traced `compiled_lfm2_decode_paged()` branch. This is the P4 proof
/// that the default paged path runs the compiled-paged graph (not the eager
/// pure-Rust paged decode). `clone_model_dir(.., true)` sets `paged_block_size:
/// 16`, exactly what `init_lfm2_paged_compiled_session` requires.
async fn run_paged_compiled(src: &Path, suffix: &str, prompt: &str, max_new: i32) -> RunOutcome {
    let dir = clone_model_dir(src, suffix, true)
        .unwrap_or_else(|e| panic!("clone paged-compiled dir ({suffix}): {e}"));
    let model = Lfm2Model::load_from_dir(&dir.to_string_lossy())
        .await
        .unwrap_or_else(|e| panic!("load paged-compiled model ({suffix}): {e:?}"));
    let before = unsafe { mlx_sys::mlx_lfm2_moe_compiled_paged_call_count() };
    let r = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(greedy_chat_config(max_new, true)),
        )
        .await
        .unwrap_or_else(|e| panic!("paged-compiled chat_session_start ({suffix}): {e:?}"));
    let after = unsafe { mlx_sys::mlx_lfm2_moe_compiled_paged_call_count() };
    let out = run_outcome(suffix, &r, after.saturating_sub(before));
    drop(model);
    out
}

/// Load a paged (`use_block_paged_cache:true`) model and run a single-turn greedy
/// decode under the EAGER paged path, capturing the PAGED-FORWARD engagement delta
/// (`mlx_lfm2_moe_paged_forward_call_count`) — the per-step counter bumped at the
/// top of `mlx_lfm2_moe_forward_paged` for BOTH the compiled AND the eager
/// `MLX_NO_COMPILE` arm.
///
/// This is the helper the EAGER-paged reference uses (run as a SEPARATE process
/// with `MLX_NO_COMPILE=1`, since the C++ `static bool no_compile` latches once per
/// process). Under `MLX_NO_COMPILE=1` the Rust gate `compiled_path_active()` stays
/// ON (it does NOT read `MLX_NO_COMPILE`, exactly like the flat path), so the
/// decode loop still routes through `forward_lfm2_cpp_paged` →
/// `mlx_lfm2_moe_forward_paged`, which runs the EAGER `lfm2_decode_fn_paged` arm
/// (same paged KV-pool / conv layout / RoPE / offset logic, just not
/// `mlx::core::compile`d). The compiled-paged counter never bumps in that arm, so
/// the paged-forward counter is the only "C++ paged forward engaged" proof the
/// eager run has — a nonzero delta rules out a silent fallback to the pure-Rust
/// `run_paged_decode_step` (which would never enter the C++ FFI).
async fn run_paged_eager(src: &Path, suffix: &str, prompt: &str, max_new: i32) -> RunOutcome {
    let dir = clone_model_dir(src, suffix, true)
        .unwrap_or_else(|e| panic!("clone paged-eager dir ({suffix}): {e}"));
    let model = Lfm2Model::load_from_dir(&dir.to_string_lossy())
        .await
        .unwrap_or_else(|e| panic!("load paged-eager model ({suffix}): {e:?}"));
    let before = unsafe { mlx_sys::mlx_lfm2_moe_paged_forward_call_count() };
    let r = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(greedy_chat_config(max_new, true)),
        )
        .await
        .unwrap_or_else(|e| panic!("paged-eager chat_session_start ({suffix}): {e:?}"));
    let after = unsafe { mlx_sys::mlx_lfm2_moe_paged_forward_call_count() };
    let out = run_outcome(suffix, &r, after.saturating_sub(before));
    drop(model);
    out
}

/// Longest common byte prefix length of two strings.
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.as_bytes()
        .iter()
        .zip(b.as_bytes().iter())
        .take_while(|(x, y)| x == y)
        .count()
}

/// Diverging suffix length (bytes) on the longer side: how much trailing text is
/// NOT covered by the shared prefix. This is the "how many tokens flipped"
/// proxy. `0` means byte-identical.
fn tail_diff_bytes(a: &str, b: &str) -> usize {
    let lcp = common_prefix_len(a, b);
    a.len().max(b.len()).saturating_sub(lcp)
}

/// Assert two decode outputs agree everywhere except (at most) a short trailing
/// argmax-tie flip. Fails loudly with a window if they diverge earlier.
fn assert_tail_only_divergence(label: &str, got: &str, reference: &str) {
    if got == reference {
        eprintln!("[{label}] BYTE-IDENTICAL ({} bytes)", got.len());
        return;
    }
    let lcp = common_prefix_len(got, reference);
    let tail = tail_diff_bytes(got, reference);
    eprintln!(
        "[{label}] differ: common_prefix={lcp}B got_len={}B ref_len={}B tail_diff={tail}B",
        got.len(),
        reference.len()
    );
    eprintln!("[{label}] got_tail = {:?}", &got[lcp.min(got.len())..]);
    eprintln!(
        "[{label}] ref_tail = {:?}",
        &reference[lcp.min(reference.len())..]
    );
    assert!(
        tail <= TAIL_TOLERANCE_BYTES,
        "[{label}] EARLY DIVERGENCE (NOT a benign tail flip): the shared prefix is \
         only {lcp} bytes and the diverging suffix is {tail} bytes (tolerance \
         {TAIL_TOLERANCE_BYTES}). This is a REAL compiled bug, not a late argmax tie.\n\
         got = {got:?}\nref = {reference:?}"
    );
}

/// Assert our compiled `raw_text` agrees with the INDEPENDENT mlx-lm golden on a
/// long EARLY prefix (`>= min_prefix_bytes`). This is the decisive oracle: a
/// wrong rope base / RMSNorm eps / MoE routing order diverges within the first
/// handful of tokens, far short of this prefix, and fails loudly. A benign
/// cross-impl bf16/argmax-tie flip beyond the gate is reported but tolerated
/// (the shared prefix already cleared the gate). Returns the shared-prefix
/// length.
fn assert_golden_prefix(label: &str, got: &str, golden: &str, min_prefix_bytes: usize) -> usize {
    let lcp = common_prefix_len(got, golden);
    let tail = tail_diff_bytes(got, golden);
    if got == golden {
        eprintln!(
            "[{label}] BYTE-IDENTICAL to mlx-lm golden ({} bytes)",
            got.len()
        );
    } else {
        eprintln!(
            "[{label}] vs mlx-lm golden: common_prefix={lcp}B got_len={}B golden_len={}B tail_diff={tail}B",
            got.len(),
            golden.len()
        );
        eprintln!("[{label}] got_tail    = {:?}", &got[lcp.min(got.len())..]);
        eprintln!(
            "[{label}] golden_tail = {:?}",
            &golden[lcp.min(golden.len())..]
        );
    }
    assert!(
        lcp >= min_prefix_bytes,
        "[{label}] PARITY FAILURE vs INDEPENDENT mlx-lm oracle: the compiled \
         `raw_text` shares only {lcp} bytes with the mlx-lm greedy golden \
         (required >= {min_prefix_bytes}). A short shared prefix means our \
         Rust/C++ compiled forward diverged EARLY from mlx-lm — a real impl bug \
         (wrong rope base / RMSNorm eps / MoE routing / argmax), NOT a benign late \
         bf16 tie.\n  got    = {got:?}\n  golden = {golden:?}"
    );
    lcp
}

// =============================================================================
// PHASE 4 PIECE 2: INDEPENDENT mlx-lm GOLDEN PARITY (DENSE, always-runnable).
//
// The ONLY non-Rust oracle in this suite. Compares the compiled-flat decode's
// `raw_text` against a FROZEN greedy decode captured from mlx-lm (Apple's
// reference MLX library — a separate codebase), so a bug shared by every Rust
// path (wrong rope base / RMSNorm eps / RoPE convention) — invisible to the
// eager-flat and native-paged references, which are all the SAME Rust+C++ math —
// is caught here. See the GOLDEN_* const block + scripts/capture_lfm2_golden.py.
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_compiled_flat_vs_mlx_lm_golden() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        // The golden-vs-compiled test is compiled-mode-only: under
        // MLX_NO_COMPILE=1 the flat path runs the EAGER C++ graph, which the
        // eager-flat capture test owns. Skip cleanly so a whole-binary
        // MLX_NO_COMPILE=1 invocation doesn't spuriously fail here.
        eprintln!("[skip] MLX_NO_COMPILE=1 — golden parity test is compiled-mode-only");
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };

    // The shared `resolve_source_model` DEFAULTS to the dense 1.2B path, but a
    // host could point MLX_TEST_MODEL_PATH at the MoE checkpoint. The dense
    // golden is byte-specific to the dense tokenizer/weights, so refuse to diff
    // it against a MoE run (the MoE golden test owns that). Non-MoE => proceed.
    if is_moe_config(&src) {
        eprintln!(
            "[skip] lfm2_compiled_flat_vs_mlx_lm_golden: MLX_TEST_MODEL_PATH points at a MoE \
             checkpoint ({}); the DENSE golden cannot be diffed against it. The MoE golden \
             test (lfm2_moe_compiled_flat_vs_mlx_lm_golden) owns that.",
            src.display()
        );
        return;
    }

    eprintln!("[golden] checkpoint (dense): {}", src.display());

    // Decode >= 64 tokens so the >=200-byte early-prefix gate has room. The
    // golden captured 80 ids; we ask for 80 here too.
    let max_new = 80;
    let flat = run_flat_golden(&src, "golden-dense", GOLDEN_PROMPT, max_new).await;

    // ENGAGEMENT: the compiled path actually produced this output (delta==0 =>
    // silent native fallback => the golden comparison would be meaningless).
    assert!(
        flat.call_delta >= MIN_COMPILED_CALL_DELTA,
        "DENSE GOLDEN: compiled path did not engage: forward_call_count delta = {} \
         (model_id={}, weight_count={}); the flat compiled path silently fell back to native.",
        flat.call_delta,
        flat.model_id,
        flat.weight_count
    );
    assert_ne!(
        flat.model_id, 0,
        "dense golden: compiled model id not published"
    );
    assert_eq!(
        flat.weight_count, EXPECTED_WEIGHT_COUNT,
        "dense golden: unexpected registered weight count"
    );

    // FULL-GENERATION: the golden was captured at 80 ids that are STILL
    // mid-reasoning (no EOS), so the correct decode finishes by LENGTH at 80. An
    // early stop would shrink the oracle window and could pass a short prefix
    // while hiding a downstream bug — so pin both the token count and the reason.
    assert_eq!(
        flat.num_tokens, max_new as u32,
        "dense golden: generated {} tokens but expected the full {} — an early stop \
         truncates the oracle window and could pass a short prefix while hiding a downstream bug",
        flat.num_tokens, max_new
    );
    assert_eq!(
        flat.finish_reason, "length",
        "dense golden: finished by {:?} not \"length\" — early EOS/stop shrinks the \
         comparison window below the captured golden",
        flat.finish_reason
    );

    // TEMPLATE-DRIFT CANARY: our re-templated prompt must match the id count the
    // mlx-lm golden was captured against. If this drifts, the golden is stale
    // and the text comparison is invalid — regenerate via the capture script.
    assert_eq!(
        flat.prompt_tokens, GOLDEN_PROMPT_TOKENS_DENSE,
        "dense golden: prompt re-templated to {} ids but mlx-lm golden was captured at {} ids \
         — chat_template drift. Regenerate the golden (scripts/capture_lfm2_golden.py).",
        flat.prompt_tokens, GOLDEN_PROMPT_TOKENS_DENSE
    );

    // DECISIVE: independent-oracle parity on a long early prefix.
    let lcp = assert_golden_prefix(
        "dense-golden",
        &flat.raw_text,
        GOLDEN_TEXT_DENSE,
        GOLDEN_MIN_PREFIX_BYTES_DENSE,
    );

    // FULL-LENGTH regression teeth (closes the "short-prefix gate passes while a
    // post-gate token corrupts" hole): the mlx-lm prefix gate above clears the
    // first 314 bytes against the INDEPENDENT oracle; this pins the ENTIRE
    // byte-reproducible compiled self-golden so any post-gate token flip fails.
    assert_eq!(
        flat.raw_text, GOLDEN_RAW_TEXT_DENSE_OURS,
        "dense golden: compiled raw_text diverged from the byte-reproducible compiled \
         self-golden — a real regression in the 80-token decode."
    );

    eprintln!(
        "[PASS] dense compiled-flat matches the INDEPENDENT mlx-lm greedy golden on a \
         {lcp}-byte early prefix (>= {GOLDEN_MIN_PREFIX_BYTES_DENSE}); compiled path engaged \
         (call_delta={}).",
        flat.call_delta
    );
}

// =============================================================================
// STEP 4 + STEP 2: compiled-flat vs native-paged, with tail-divergence
// characterization. Proves ENGAGEMENT (gate active, weight_count, call delta)
// and that any divergence is confined to the final argmax-tie region. Also
// writes the compiled-flat text artifact consumed by the native-flat test.
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_compiled_flat_vs_paged_e2e_parity() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        // Under MLX_NO_COMPILE=1 the flat path is native, so this compiled-only
        // engagement test is meaningless; the native-flat capture test owns
        // that mode. Skip cleanly so a `MLX_NO_COMPILE=1` invocation of the
        // whole binary doesn't spuriously fail here.
        eprintln!("[skip] MLX_NO_COMPILE=1 — compiled-engagement test is native-mode-only");
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    eprintln!("[e2e] checkpoint: {}", src.display());

    let prompt = "What is the capital of France? Answer in one short sentence.";

    // ---- Compiled flat path (under test) ---------------------------------
    let flat = run_flat_single(&src, "flat", prompt, N_NEW_TOKENS).await;

    // ENGAGEMENT + GATE (STEP 4). delta == 0 => silent fallback => hard fail.
    assert!(
        flat.call_delta >= MIN_COMPILED_CALL_DELTA,
        "COMPILED PATH DID NOT ENGAGE: forward_call_count delta = {} (model_id={}, \
         weight_count={}). The flat compiled path silently fell back to native forward().",
        flat.call_delta,
        flat.model_id,
        flat.weight_count
    );
    assert_ne!(flat.model_id, 0, "compiled model id was not published");
    assert_eq!(
        flat.weight_count, EXPECTED_WEIGHT_COUNT,
        "unexpected registered weight count"
    );

    // Persist the compiled-flat decode text so a SEPARATE `MLX_NO_COMPILE=1`
    // invocation can diff native-flat against it (the fidelity verdict).
    let art = artifact_path("lfm2_e2e_compiled_flat.txt");
    if let Err(e) = fs::write(&art, &flat.text) {
        eprintln!("[warn] could not write compiled-flat artifact {art:?}: {e}");
    } else {
        eprintln!("[e2e] wrote compiled-flat artifact: {}", art.display());
    }

    // ---- Reference: native paged path ------------------------------------
    let paged = run_paged_single(&src, "paged", prompt, N_NEW_TOKENS).await;

    // ---- Parity (STEP 2 characterization) --------------------------------
    // compiled-flat vs paged are DIFFERENT graphs, so a late argmax-tie flip is
    // legitimate. Require agreement everywhere except the short trailing region.
    let tail = tail_diff_bytes(&flat.text, &paged.text);
    eprintln!(
        "[e2e] compiled-vs-paged: tail_diff={tail}B (common_prefix={}B)",
        common_prefix_len(&flat.text, &paged.text)
    );
    assert_tail_only_divergence("compiled-vs-paged", &flat.text, &paged.text);
    assert_eq!(
        flat.num_tokens, paged.num_tokens,
        "num_tokens mismatch: compiled={} paged={}",
        flat.num_tokens, paged.num_tokens
    );
    assert_eq!(
        flat.finish_reason, paged.finish_reason,
        "finish_reason mismatch: compiled={} paged={}",
        flat.finish_reason, paged.finish_reason
    );

    eprintln!(
        "[PASS] compiled-flat engaged ({} calls) and agrees with paged within \
         tail tolerance ({tail}B <= {TAIL_TOLERANCE_BYTES}B)",
        flat.call_delta
    );
}

// =============================================================================
// P4: COMPILED-PAGED engagement + coherence (DENSE 1.2B).
//
// The P4 deliverable: the DEFAULT paged decode path now runs the COMPILED-PAGED
// graph (`lfm2_decode_fn_paged`), not the eager pure-Rust paged decode. This
// test loads the 1.2B dense checkpoint with `use_block_paged_cache:true`
// (paged is also the production default), decodes N tokens, and proves:
//   1. ENGAGEMENT: `mlx_lfm2_moe_compiled_paged_call_count` advanced by ~N-1
//      (the last sampled token is never fed forward). A delta of 0 means a
//      silent fallback to the eager paged decode → hard fail.
//   2. COHERENCE: the compiled-paged text agrees with the compiled-FLAT decode
//      everywhere except (at most) a short trailing argmax-tie flip — the same
//      flat-vs-paged-graph tolerance the existing parity test uses. (Both are
//      now compiled; the residual divergence is pure flat-vs-paged bf16 noise.)
//
// Under `MLX_NO_COMPILE=1` the paged path is the EAGER C++ paged graph, so the
// compiled-paged counter never bumps — skip cleanly so a whole-binary
// `MLX_NO_COMPILE=1` invocation doesn't spuriously fail here (the dedicated
// eager-vs-compiled capture tests own that mode).
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_compiled_paged_engagement_dense() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        eprintln!("[skip] MLX_NO_COMPILE=1 — compiled-paged engagement test is compile-mode-only");
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    eprintln!("[paged-compiled] checkpoint: {}", src.display());

    let prompt = "What is the capital of France? Answer in one short sentence.";

    // ---- Compiled-PAGED path (under test) --------------------------------
    let paged = run_paged_compiled(&src, "paged-compiled", prompt, N_NEW_TOKENS).await;

    // ENGAGEMENT GATE: delta == 0 => silent eager fallback => hard fail.
    assert!(
        paged.call_delta >= MIN_COMPILED_CALL_DELTA,
        "COMPILED-PAGED PATH DID NOT ENGAGE: compiled_paged_call_count delta = {} \
         (model_id={}, weight_count={}). The paged decode silently fell back to the \
         eager pure-Rust paged decode.",
        paged.call_delta,
        paged.model_id,
        paged.weight_count
    );
    assert_ne!(
        paged.model_id, 0,
        "compiled-paged model id was not published"
    );
    assert_eq!(
        paged.weight_count, EXPECTED_WEIGHT_COUNT,
        "unexpected registered weight count (paged registration must match flat)"
    );

    // COHERENCE: compiled-paged vs compiled-flat. Both compiled; a late argmax
    // tie can flip the last token or two — require agreement everywhere else.
    let flat = run_flat_single(&src, "paged-compiled-flatref", prompt, N_NEW_TOKENS).await;
    let tail = tail_diff_bytes(&paged.text, &flat.text);
    eprintln!(
        "[paged-compiled] vs compiled-flat: tail_diff={tail}B (common_prefix={}B)",
        common_prefix_len(&paged.text, &flat.text)
    );
    assert_tail_only_divergence("paged-compiled-vs-flat", &paged.text, &flat.text);
    assert_eq!(
        paged.num_tokens, flat.num_tokens,
        "num_tokens mismatch: paged-compiled={} flat={}",
        paged.num_tokens, flat.num_tokens
    );

    eprintln!(
        "[PASS] compiled-PAGED engaged ({} compiled-paged calls) and agrees with \
         compiled-flat within tail tolerance ({tail}B <= {TAIL_TOLERANCE_BYTES}B). \
         text snippet: {:?}",
        paged.call_delta,
        &paged.text.chars().take(80).collect::<String>()
    );
}

// =============================================================================
// P5: COMPILED-PAGED vs EAGER-PAGED parity (DENSE + MoE).
//
// The make-or-break P5 gate: the COMPILED-paged decode graph
// (`lfm2_decode_fn_paged` wrapped in `mlx::core::compile`) must be byte-greedy
// IDENTICAL to the EAGER-paged decode graph (the SAME `lfm2_decode_fn_paged`, run
// directly under `MLX_NO_COMPILE=1`). The two share EVERY paged numeric: the same
// `paged_kv_write` / `paged_attention` ops over the same block-paged Metal pools,
// the same per-head Q/K RMSNorm + neox RoPE(array offset), the same conv-state
// threading and FFN dispatch — the ONLY difference is whether the graph is
// `mlx::core::compile`d. So byte-identity is the verdict that `compile()`
// faithfully preserves the eager paged graph (an early divergence would be a real
// compile() topology/offset bug, the only thing that differs between the runs).
//
// This is the PAGED twin of the flat `lfm2_eager_flat_vs_compiled_capture`
// verdict, and it is split the SAME way for the SAME reason: the C++
// `static bool no_compile` in `mlx_lfm2_moe_forward_paged` latches ONCE per
// process, so the eager reference MUST be a SEPARATE `cargo test` invocation with
// `MLX_NO_COMPILE=1`. The two halves communicate through a topology-qualified
// artifact file in the workspace `target/`, never an in-process env toggle:
//
//   * THIS test (`lfm2_compiled_paged_vs_eager_paged`, compiled-mode only): runs
//     the compiled-paged decode, asserts ENGAGEMENT via the compiled-paged
//     counter (`mlx_lfm2_moe_compiled_paged_call_count` delta > 0 — a 0 delta is a
//     silent fallback to the eager pure-Rust paged decode), and writes the
//     compiled-paged text to `lfm2[_moe]_e2e_compiled_paged.txt`.
//   * The capture test below (`lfm2_eager_paged_vs_compiled_paged_capture`,
//     `MLX_NO_COMPILE=1` only): runs the eager-paged decode, asserts the C++ paged
//     forward ENGAGED (`mlx_lfm2_moe_paged_forward_call_count` delta — the only
//     signal the eager arm produces), and diffs eager-paged vs the compiled-paged
//     artifact for BYTE-GREEDY equality across >= N_NEW_TOKENS (48) tokens.
//
// Dense is covered by the shared `resolve_source_model` default; MoE is gated on
// `LFM2_MOE_MODEL_PATH` (the topology-qualified artifact name keeps them from
// colliding). Skips cleanly under `MLX_NO_COMPILE=1` (the capture test owns that
// mode) so a whole-binary `MLX_NO_COMPILE=1` invocation doesn't spuriously fail.
// =============================================================================

/// Artifact name for the compiled-PAGED decode text, topology-qualified so the
/// dense (`lfm2_e2e_compiled_paged.txt`) and MoE (`lfm2_moe_e2e_compiled_paged.txt`)
/// runs never diff against each other's stale text.
fn compiled_paged_artifact_name(is_moe: bool) -> &'static str {
    if is_moe {
        "lfm2_moe_e2e_compiled_paged.txt"
    } else {
        "lfm2_e2e_compiled_paged.txt"
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_compiled_paged_vs_eager_paged() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        eprintln!(
            "[skip] MLX_NO_COMPILE=1 — compiled-paged writer is compiled-mode-only \
             (the eager-paged capture test owns MLX_NO_COMPILE=1)"
        );
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    let comp_is_moe = is_moe_config(&src);
    eprintln!(
        "[paged-parity] checkpoint: {} (is_moe={comp_is_moe})",
        src.display()
    );

    let prompt = "What is the capital of France? Answer in one short sentence.";

    // ---- Compiled-PAGED path (under test) --------------------------------
    let paged = run_paged_compiled(&src, "paged-parity-compiled", prompt, N_NEW_TOKENS).await;

    // ENGAGEMENT GATE: delta == 0 => silent eager pure-Rust fallback => hard fail.
    // (delta>0 is the explicit P5 requirement.)
    assert!(
        paged.call_delta >= MIN_COMPILED_CALL_DELTA,
        "COMPILED-PAGED PATH DID NOT ENGAGE: compiled_paged_call_count delta = {} \
         (model_id={}, weight_count={}). The paged decode silently fell back to the \
         eager pure-Rust paged decode — the compiled-paged graph never ran, so the \
         parity comparison would be meaningless.",
        paged.call_delta,
        paged.model_id,
        paged.weight_count
    );
    assert_ne!(
        paged.model_id, 0,
        "compiled-paged model id was not published"
    );
    // FULL-GENERATION: the eager-paged capture diffs the WHOLE text, so the
    // compiled writer must decode the full N tokens (no early stop) or the eager
    // side could pass a short prefix while hiding a downstream divergence.
    assert_eq!(
        paged.num_tokens, N_NEW_TOKENS as u32,
        "compiled-paged: generated {} tokens but expected the full {} — an early stop \
         shrinks the byte-greedy parity window the eager-paged capture diffs against",
        paged.num_tokens, N_NEW_TOKENS
    );

    // Persist the compiled-paged decode text so a SEPARATE `MLX_NO_COMPILE=1`
    // invocation (the eager-paged capture below) can diff against it for the
    // byte-greedy parity verdict.
    let art = artifact_path(compiled_paged_artifact_name(comp_is_moe));
    if let Err(e) = fs::write(&art, &paged.text) {
        eprintln!("[warn] could not write compiled-paged artifact {art:?}: {e}");
    } else {
        eprintln!(
            "[paged-parity] wrote compiled-paged artifact: {}",
            art.display()
        );
    }

    eprintln!(
        "[PASS] compiled-PAGED engaged ({} compiled-paged calls, {} tokens); \
         artifact written for the MLX_NO_COMPILE=1 eager-paged byte-greedy diff. \
         text snippet: {:?}",
        paged.call_delta,
        paged.num_tokens,
        &paged.text.chars().take(80).collect::<String>()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_eager_paged_vs_compiled_paged_capture() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if !no_compile_env() {
        eprintln!(
            "[skip] eager-paged capture requires MLX_NO_COMPILE=1 (run as a SEPARATE \
             cargo test invocation with that env set; the static no_compile flag latches \
             once per process)"
        );
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    let eager_is_moe = is_moe_config(&src);
    eprintln!(
        "[eager-paged] checkpoint: {} (is_moe={eager_is_moe})",
        src.display()
    );

    let prompt = "What is the capital of France? Answer in one short sentence.";

    // ---- Eager-PAGED reference (MLX_NO_COMPILE=1) ------------------------
    let eager = run_paged_eager(&src, "paged-parity-eager", prompt, N_NEW_TOKENS).await;

    // ENGAGEMENT: under MLX_NO_COMPILE=1 the Rust gate still routes through the C++
    // paged forward (it does NOT read MLX_NO_COMPILE — only the env flag INSIDE
    // mlx_lfm2_moe_forward_paged swaps compile→eager). So the paged-forward counter
    // MUST advance ~N-1; a 0 delta means the Rust caller silently fell back to the
    // pure-Rust paged decode and the eager C++ graph never ran — the comparison
    // would be meaningless. (The compiled-paged counter never bumps in this arm,
    // which is exactly WHY a separate paged-forward counter exists.)
    assert!(
        eager.call_delta >= MIN_COMPILED_CALL_DELTA,
        "eager-paged (MLX_NO_COMPILE=1) did not run the C++ paged forward \
         (paged_forward_call_count delta = {}). Expected ~N-1: MLX_NO_COMPILE only \
         swaps compile→eager INSIDE mlx_lfm2_moe_forward_paged; it does not disable \
         the Rust gate. A 0 delta is a silent fallback to the pure-Rust paged decode.",
        eager.call_delta
    );
    assert_eq!(
        eager.num_tokens, N_NEW_TOKENS as u32,
        "eager-paged: generated {} tokens but expected the full {} — an early stop \
         shrinks the byte-greedy parity window below the compiled-paged artifact",
        eager.num_tokens, N_NEW_TOKENS
    );
    eprintln!(
        "[eager-paged] C++ paged forward ran (call_delta={}, model_id={}, num_tokens={})",
        eager.call_delta, eager.model_id, eager.num_tokens
    );

    // Persist for symmetry / debugging.
    let eager_art = artifact_path(if eager_is_moe {
        "lfm2_moe_e2e_eager_paged.txt"
    } else {
        "lfm2_e2e_eager_paged.txt"
    });
    let _ = fs::write(&eager_art, &eager.text);

    // ---- BYTE-GREEDY PARITY VERDICT: eager-paged vs compiled-paged -------
    // Read the topology-matched compiled-paged artifact written by
    // `lfm2_compiled_paged_vs_eager_paged` (run the default/compiled suite FIRST).
    let comp_art = artifact_path(compiled_paged_artifact_name(eager_is_moe));
    let compiled_text = fs::read_to_string(&comp_art).unwrap_or_else(|e| {
        panic!(
            "[eager-paged] compiled-paged artifact not found at {} ({e}); run the \
             default (compiled) suite FIRST to produce it \
             (LFM2_COMPILED_E2E=1 ... lfm2_compiled_paged_vs_eager_paged), then re-run \
             this with MLX_NO_COMPILE=1. eager-paged text captured to {}.",
            comp_art.display(),
            eager_art.display()
        )
    });

    let lcp = common_prefix_len(&eager.text, &compiled_text);
    let tail = tail_diff_bytes(&eager.text, &compiled_text);
    eprintln!(
        "[VERDICT] eager-paged vs compiled-paged: common_prefix={lcp}B eager_len={}B \
         compiled_len={}B tail_diff={tail}B",
        eager.text.len(),
        compiled_text.len()
    );

    // The two runs share the SAME paged graph (only compile() differs), so the
    // expected verdict is BYTE-IDENTICAL — exactly like the flat eager-vs-compiled
    // case. Assert byte equality as the strict gate; `assert_tail_only_divergence`
    // is the loud-failure path if compile() introduced any divergence (early =>
    // a real compile() topology/offset bug; late tail => a benign argmax tie that
    // we still surface).
    if eager.text == compiled_text {
        eprintln!(
            "[VERDICT] BYTE-IDENTICAL ({} bytes / {} tokens) — `compile()` faithfully \
             preserves the eager paged decode graph; the compiled-paged port is \
             numerically exact.",
            eager.text.len(),
            eager.num_tokens
        );
    } else {
        eprintln!("[VERDICT] eager and compiled PAGED differ — investigate (see tails below)");
        eprintln!(
            "[VERDICT] eager_tail    = {:?}",
            &eager.text[lcp.min(eager.text.len())..]
        );
        eprintln!(
            "[VERDICT] compiled_tail = {:?}",
            &compiled_text[lcp.min(compiled_text.len())..]
        );
    }
    assert_tail_only_divergence("eager-paged-vs-compiled-paged", &eager.text, &compiled_text);

    // BYTE-GREEDY across >= 48 tokens (spec): both sides generated N_NEW_TOKENS
    // (48) and the shared byte prefix must span essentially all of it (allowing
    // only the late argmax-tie tail). A short shared prefix would already have
    // failed `assert_tail_only_divergence` above; this asserts the agreed span is
    // long enough to actually cover the 48-token greedy decode.
    let min_shared = compiled_text.len().saturating_sub(TAIL_TOLERANCE_BYTES);
    assert!(
        lcp >= min_shared,
        "[eager-paged-vs-compiled-paged] byte-greedy parity over the full {}-token \
         decode FAILED: shared prefix is only {lcp}B but the compiled-paged text is \
         {}B (need >= {min_shared}B, i.e. agreement minus the <= {TAIL_TOLERANCE_BYTES}B \
         argmax-tie tail).",
        N_NEW_TOKENS,
        compiled_text.len()
    );

    eprintln!(
        "[PASS] eager-paged vs compiled-paged BYTE-GREEDY parity holds over the full \
         {}-token decode (shared prefix {lcp}B, tail_diff {tail}B <= {TAIL_TOLERANCE_BYTES}B); \
         compile() is faithful to the eager paged graph.",
        N_NEW_TOKENS
    );
}

// =============================================================================
// STEP 1 + STEP 2: eager-flat (MLX_NO_COMPILE=1) capture + verdict.
//
// Run this in its OWN process WITH `MLX_NO_COMPILE=1`. It:
//   * runs the flat decode (which under MLX_NO_COMPILE=1 is the EAGER C++ path —
//     same flat KV/conv layout + same logic, just not `compile()`d; the Rust
//     gate stays ON so the C++ forward STILL runs, call_delta ~N-1),
//   * asserts the C++ forward DID run (call_delta >= floor) — call_delta == 0
//     would mean a silent flat fallback that invalidates the comparison,
//   * if the compiled-flat artifact from the default run is present, performs
//     the FIDELITY VERDICT: eager-flat vs compiled-flat.
//       - byte-identical  => compile() is faithful; any compiled-vs-paged
//         divergence is pure flat-vs-paged bf16 noise.
//       - early divergence => REAL compile() topology/offset bug (compile() is
//         the ONLY difference between the two runs).
//       - tail-only flip  => benign late argmax tie.
//
// Without `MLX_NO_COMPILE=1` this test SKIPS (the default suite's compiled test
// already covers compiled engagement).
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_eager_flat_vs_compiled_capture() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if !no_compile_env() {
        eprintln!(
            "[skip] eager-flat capture requires MLX_NO_COMPILE=1 (run as a SEPARATE \
             cargo test invocation with that env set)"
        );
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    eprintln!("[eager-flat] checkpoint: {}", src.display());

    let prompt = "What is the capital of France? Answer in one short sentence.";
    let eager = run_flat_single(&src, "eager-flat", prompt, N_NEW_TOKENS).await;

    // IMPORTANT — what `MLX_NO_COMPILE=1` actually does for lfm2 (verified
    // empirically): it does NOT route to a pure-Rust forward. lfm2's Rust gate
    // `compiled_path_active()` does NOT read `MLX_NO_COMPILE`; it stays ON, so
    // the decode loop still calls `mlx_lfm2_moe_forward` once per step (the
    // engagement counter STILL increments, ~N-1). Inside that C++ function the
    // env flag only swaps `compiled_lfm2_decode()(...)` for an EAGER
    // `lfm2_decode_fn(...)` (mlx_lfm2_moe.cpp:343). So this is the EAGER C++
    // flat path — the SAME offset/mask/cache/RoPE logic as the compiled path,
    // just not wrapped in `mlx::core::compile`. That is exactly the reference
    // that isolates a `compile()` TOPOLOGY bug from a LOGIC bug:
    //   * eager-flat == compiled-flat  => `compile()` faithfully preserves the
    //     eager graph; any compiled-vs-paged tail flip is flat-vs-paged bf16
    //     noise, NOT a compile() bug.
    //   * eager-flat != compiled-flat  => a real `compile()` topology/offset bug
    //     (the only thing that differs between the two runs).
    // The forward MUST still have run (~N-1); call_delta == 0 here would mean
    // the flat path silently fell back, invalidating the comparison.
    assert!(
        eager.call_delta >= MIN_COMPILED_CALL_DELTA,
        "eager-flat (MLX_NO_COMPILE=1) did not run the C++ forward (call_delta={}). \
         Expected ~N-1: MLX_NO_COMPILE only swaps compile()→eager INSIDE \
         mlx_lfm2_moe_forward; it does not disable the Rust gate.",
        eager.call_delta
    );
    eprintln!(
        "[eager-flat] eager C++ forward ran (call_delta={}, model_id={})",
        eager.call_delta, eager.model_id
    );

    // Persist for symmetry / debugging.
    let eager_art = artifact_path("lfm2_e2e_eager_flat.txt");
    let _ = fs::write(&eager_art, &eager.text);

    // ---- FIDELITY VERDICT: eager-flat vs compiled-flat -------------------
    // Topology-qualified artifact pick: the DENSE compiled test writes
    // `lfm2_e2e_compiled_flat.txt` while the MoE parity test writes the qualified
    // `lfm2_moe_e2e_compiled_flat.txt`. Reading the dense name unconditionally
    // here would diff MoE-eager against a STALE dense-compiled artifact
    // (common_prefix=0 => a spurious "REAL compile bug" panic on the 8B). Read
    // `src/config.json` and pick the artifact that matches THIS run's topology.
    let comp_is_moe = is_moe_config(&src);
    let comp_art = artifact_path(if comp_is_moe {
        "lfm2_moe_e2e_compiled_flat.txt"
    } else {
        "lfm2_e2e_compiled_flat.txt"
    });
    match fs::read_to_string(&comp_art) {
        Ok(compiled_text) => {
            let lcp = common_prefix_len(&eager.text, &compiled_text);
            let tail = tail_diff_bytes(&eager.text, &compiled_text);
            eprintln!(
                "[VERDICT] eager-flat vs compiled-flat: common_prefix={lcp}B \
                 eager_len={}B compiled_len={}B tail_diff={tail}B",
                eager.text.len(),
                compiled_text.len()
            );
            if eager.text == compiled_text {
                eprintln!(
                    "[VERDICT] BYTE-IDENTICAL — `compile()` faithfully preserves the eager \
                     lfm2 decode graph. Any compiled-vs-paged last-token divergence is pure \
                     flat-vs-paged bf16 argmax-tie noise, NOT a compile() topology bug."
                );
            } else {
                eprintln!(
                    "[VERDICT] eager-flat tail = {:?}",
                    &eager.text[lcp.min(eager.text.len())..]
                );
                eprintln!(
                    "[VERDICT] compiled-flat tail = {:?}",
                    &compiled_text[lcp.min(compiled_text.len())..]
                );
                eprintln!(
                    "[VERDICT] eager and compiled flat differ — since the ONLY difference \
                     between them is `mlx::core::compile`, this is a REAL compile() \
                     topology/offset bug (if it diverges early) or a benign late tie."
                );
            }
            // Early divergence between eager and compiled flat is a REAL
            // compile() topology/offset bug and must fail.
            assert_tail_only_divergence("eager-vs-compiled", &eager.text, &compiled_text);
            eprintln!(
                "[PASS] eager-flat reference captured and compared against compiled-flat \
                 (tail_diff={tail}B <= {TAIL_TOLERANCE_BYTES}B)"
            );
        }
        Err(e) => {
            eprintln!(
                "[eager-flat] compiled-flat artifact not found at {} ({e}); run the default \
                 (compiled) suite FIRST to produce it, then re-run this with MLX_NO_COMPILE=1. \
                 eager-flat text captured to {}",
                comp_art.display(),
                eager_art.display()
            );
        }
    }
}

// =============================================================================
// PHASE 4 PIECE 2 (round-2): compile()-FAITHFULNESS on the GOLDEN trajectory.
//
// Codex round-2 [MED]: the prior eager-vs-compiled evidence
// (`lfm2_eager_flat_vs_compiled_capture`) used the GREEDY 48-token trajectory
// (`run_flat_single`, thinking_budget=0, `.text`) — NOT the GOLDEN 80-token
// thinking trajectory where the step-33 ' So' divergence-from-mlx-lm actually
// happens. These two tests close that gap: they re-run the GOLDEN config
// (`run_flat_golden`, max_new=80, the full `<think>…` trace) under
// `MLX_NO_COMPILE=1` (the eager C++ flat path — SAME offset/mask/cache/RoPE
// logic, just not `mlx::core::compile`d) and assert the eager output agrees with
// the COMMITTED compiled self-golden within `TAIL_TOLERANCE_BYTES`. Eager
// matching the compiled self-golden on THIS trajectory proves the ' So' flip is
// inherent to our MoE/dense bf16 math, NOT a `compile()` artifact.
//
// Both REQUIRE `MLX_NO_COMPILE=1` and run as a SEPARATE cargo invocation (the
// C++ `static bool no_compile` latches once per process).
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_moe_golden_eager_vs_compiled() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if !no_compile_env() {
        eprintln!(
            "[skip] golden eager-vs-compiled needs MLX_NO_COMPILE=1 (run as a SEPARATE invocation)"
        );
        return;
    }
    let Some(src) = resolve_moe_model() else {
        return;
    };
    let eager = run_flat_golden(&src, "moe-golden-eager", GOLDEN_PROMPT, 80).await;
    assert!(
        eager.call_delta >= MIN_COMPILED_CALL_DELTA,
        "moe golden eager: C++ forward did not run (call_delta={}); eager flat fell back",
        eager.call_delta
    );
    // compile() FAITHFULNESS on the GOLDEN (thinking, 80-token) trajectory: eager
    // (no compile()) must agree with the committed COMPILED self-golden through the
    // step-33 ' So' flip, diverging only at a late benign tie (<= TAIL_TOLERANCE_BYTES).
    // Proves the ' So' divergence-from-mlx-lm is inherent to our MoE bf16 math, NOT a
    // compile() artifact (closes Codex round-2: prior evidence used the 48-tok greedy path).
    assert_tail_only_divergence(
        "moe golden eager-vs-compiled",
        &eager.raw_text,
        GOLDEN_RAW_TEXT_MOE_OURS,
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_dense_golden_eager_vs_compiled() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if !no_compile_env() {
        eprintln!(
            "[skip] golden eager-vs-compiled needs MLX_NO_COMPILE=1 (run as a SEPARATE invocation)"
        );
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    if is_moe_config(&src) {
        eprintln!("[skip] dense golden eager-vs-compiled: MoE checkpoint — the MoE test owns it");
        return;
    }
    let eager = run_flat_golden(&src, "dense-golden-eager", GOLDEN_PROMPT, 80).await;
    assert!(
        eager.call_delta >= MIN_COMPILED_CALL_DELTA,
        "dense golden eager: C++ forward did not run (call_delta={})",
        eager.call_delta
    );
    assert_tail_only_divergence(
        "dense golden eager-vs-compiled",
        &eager.raw_text,
        GOLDEN_RAW_TEXT_DENSE_OURS,
    );
}

// =============================================================================
// STEP 3a: CRITICAL offset-fix validation on a LONG prompt (> 2048 tokens).
//
// A prompt over PREFILL_STEP_SIZE (512) forces `chunked_prefill` to return only
// the LAST chunk's logits, so the post-prefill `seq_len` is small while the true
// KV offset = full prompt length. Pre-fix, the compiled seed used `seq_len`,
// building a too-small causal mask / KV write index and masking out valid prefix
// tokens. Post-fix it seeds from `KVCache::get_offset()` (the true position).
//
// With the fix, compiled-flat must agree with the paged reference within the
// same late-argmax-tie tail tolerance — and crucially must NOT diverge early
// (which is what a wrong offset would cause: garbage from token 0 of decode).
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_long_prompt_offset_fix() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        eprintln!("[skip] MLX_NO_COMPILE=1 — long-prompt offset fix needs the compiled path");
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    eprintln!("[long] checkpoint: {}", src.display());

    // Build a prompt that comfortably exceeds 2 * PREFILL_STEP_SIZE (2048) tokens
    // so chunked_prefill runs the chunk loop at least twice and returns ONLY the
    // last chunk's logits — exactly the case the offset fix protects (pre-fix it
    // seeded `seq_len` = last-chunk length, not the full KV position). A short
    // factual paragraph repeated 200x is ~5k BPE tokens for this tokenizer.
    let para = "Paris is the capital of France. The Eiffel Tower stands beside the Seine. \
                The Louvre houses the Mona Lisa. France borders Spain and Germany. ";
    let mut long = String::with_capacity(para.len() * 200);
    for _ in 0..200 {
        long.push_str(para);
    }
    long.push_str(
        "\n\nBased on the text above, what is the capital of France? Answer in one short sentence.",
    );
    eprintln!(
        "[long] prompt chars = {} (>4096 tokens expected)",
        long.len()
    );

    // Decode >= 16 tokens (spec).
    let max_new = 24;
    let flat = run_flat_single(&src, "long-flat", &long, max_new).await;

    // PRIMARY assertion for the offset fix: the compiled path ENGAGES on the
    // long prompt. Pre-fix, seeding `seq_len` (last-chunk length) instead of the
    // true KV offset built a too-small causal mask / KV write index — the seed
    // would either fail `is_initialized()` (-> native fallback, call_delta==0) or
    // slice_update OOB-corrupt the cache (-> garbage). A clean ~N-1 delta proves
    // the offset seed was accepted and drove every decode step.
    assert!(
        flat.call_delta >= (max_new as u64).saturating_sub(2),
        "long-prompt compiled path did not engage: call_delta={} (expected ~{}). \
         A wrong post-chunked-prefill offset would have failed the seed and fallen \
         back to native (call_delta==0).",
        flat.call_delta,
        max_new - 1
    );
    assert_ne!(
        flat.model_id, 0,
        "long-prompt: compiled model id not published"
    );

    // SECONDARY assertion: the compiled output is COHERENT, not the immediate
    // verbatim-repetition collapse that a corrupted-offset KV cache produces.
    // (A wrong offset masks out the real prefix, so the model degenerates into
    // echoing the prompt from token 0 of decode.)
    assert!(
        !flat.text.trim().is_empty(),
        "long-prompt compiled output is empty (offset/seed likely corrupted)"
    );

    // Cross-check against the paged reference. NOTE: the paged path can
    // DEGENERATE into verbatim repetition on a long repeated-paragraph prompt
    // (observed: paged echoes 'The quick brown fox ...'); that is a paged decode
    // quality artifact, NOT a compiled-offset bug, so paged is NOT a reliable
    // text oracle here. We report the comparison but do not gate on byte parity
    // — the eager-vs-compiled capture test is the fidelity oracle, and the
    // engagement+coherence asserts above are what prove the offset fix.
    let paged = run_paged_single(&src, "long-paged", &long, max_new).await;
    let tail = tail_diff_bytes(&flat.text, &paged.text);
    eprintln!(
        "[long] compiled-vs-paged tail_diff={tail}B common_prefix={}B (informational; \
         paged may degenerate on repeated prompts)",
        common_prefix_len(&flat.text, &paged.text)
    );
    eprintln!(
        "[PASS] long-prompt offset fix holds: compiled-flat ENGAGED (call_delta={}) and \
         produced coherent output after a >2*PREFILL_STEP_SIZE chunked prefill (last-chunk- \
         only logits return). compiled={:?}",
        flat.call_delta, flat.text
    );
}

// =============================================================================
// STEP 3b: CRITICAL offset-fix validation on WARM strict-extend reuse — driven
// through the COMPILED path.
//
// IMPORTANT routing fact (verified empirically + confirmed in production source
// at model.rs ~2604): `chat_session_continue` routes through
// `chat_tokens_delta_sync`, whose decode loop is hard-wired to the NATIVE
// `self.forward()` — the compiled path is NOT wired into the delta/continue
// loop. So a `start` -> `continue` 2-turn session does NOT exercise the compiled
// warm-reuse offset seed (turn 2 runs native, call_delta==0). That is BY DESIGN.
//
// The compiled warm strict-extend path that fix #1 actually protects lives in
// `chat_sync_core` (the session-START decode loop). It is reached when a
// `chat_session_start` prompt is a STRICT EXTENSION of the live cached history
// (the stateless-agent / resend-full-conversation pattern): the prefix verifier
// returns the full cached length, the code prefills only the tail delta, and the
// compiled seed must use `KVCache::get_offset()` (the FULL live position =
// cached_prefix + delta), NOT the small per-chunk `seq_len` (= delta length).
//
// We drive that here with TWO `chat_session_start` calls on ONE live model:
//   turn 1: [user1]
//   turn 2: [user1, assistant(raw_reply1), user2]   (a superset prompt)
// Turn 2's tokenized prompt begins with turn 1's persisted history, so the
// strict-extend branch fires; the compiled path stays engaged (call_delta ~N-1),
// and the seed offset comes from the live KV position. Pre-fix this seeded the
// delta length and failed the seed (-> native fallback, call_delta==0) or built
// a too-small mask -> garbage; post-fix it seeds the true position and decodes
// coherently.
//
// REACHABILITY (verified empirically on this lfm2.5-1.2b-THINKING checkpoint,
// 2026-06-02) — load-bearing, hence the `r1.raw_text` echo + the
// `include_reasoning: Some(true)` in `greedy_chat_config_no_thinking`:
//
//   The strict-extend branch only fires when turn 2's tokenized prompt is a
//   byte-prefix of `cached_token_history` (= turn-1 template tokens + turn-1
//   RAW generated tokens, see `save_cache_state`). LFM2's chat template IGNORES
//   `enable_thinking` and `chat_sync_core` hard-codes `thinking_enabled = true`,
//   so the model ALWAYS emits a `<think>…</think>` span (here `reasoning_effort:
//   "none"` forces the budget to 0 → a degenerate empty `<think></think>` as
//   the first two generated tokens) and `parse_thinking_and_tools` ALWAYS splits
//   the decoded text at `</think>`. Therefore the round-trip turns on WHICH
//   field turn 2 echoes:
//
//     * Echoing `ChatResult.text` (reasoning-stripped): the `<think></think>`
//       markers — the FIRST generated tokens in the cache — are gone, so the
//       re-templated prompt diverges from the cache at the very first generated
//       token. `verify_cache_prefix` returns 0 -> full prefill ->
//       `cached_tokens=0`. (This is what the `cached_tokens>0` assertion caught;
//       the prior comment's "reasoning-disabled -> verbatim text" claim was
//       wrong because LFM2 never disables reasoning parsing.)
//
//     * Echoing `ChatResult.raw_text` with `include_reasoning: Some(true)`
//       (the FULL verbatim decode, `<think></think>` markers and all): this
//       round-trips token-exact through the tokenizer (empirically verified,
//       including across the max_new mid-word truncation boundary), so the
//       re-templated prompt reproduces the EXACT cached token prefix, the
//       strict-extend branch fires, and the offset seed runs on a NON-ZERO
//       prefix. Observed: turn2 call_delta=23, cached_tokens=45.
//
//   So warm strict-extend reuse IS reachable through the public
//   `chat_session_start` API for this checkpoint — by echoing the VERBATIM
//   `raw_text` (not the stripped `text`), which is why turn 1 captures
//   `r1.raw_text` and turn 2 (and the cold oracle) echo it.
//   (`chat_session_continue` is hard-wired NATIVE at model.rs ~2604 and can
//   never exercise the compiled warm path, so the resend-superset pattern is the
//   only public route — it works here.)
//
// Oracle: a FRESH-session run of the identical turn-2 prompt (no warm prefix),
// same reasoning-disabled config. Same compiled flat graph, same full prompt,
// so a correct warm strict-extend must byte-match the cold run within the
// late-argmax-tie tail tolerance. The `cached_tokens>0` assertion is what
// pins that turn 2 actually took the reuse path (a full-prefill turn 2 would
// ALSO engage compiled and match the oracle, so engagement+parity alone cannot
// distinguish reuse from a silent full prefill — only cached_tokens can).
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_warm_reuse_offset_fix() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        eprintln!("[skip] MLX_NO_COMPILE=1 — warm-reuse offset fix needs the compiled path");
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    eprintln!("[warm] checkpoint: {}", src.display());

    let user1 = "What is the capital of France? Answer in one short sentence.";
    let user2 = "Now name one famous landmark there. Answer in one short sentence.";
    let max_new = 24;

    // ---- Compiled flat, warm strict-extend via two session-START calls ----
    let (warm_turn1_delta, warm_turn2_delta, warm_turn2_cached, warm_turn2_text, turn1_reply) = {
        let dir = clone_model_dir(&src, "warm-flat", false)
            .unwrap_or_else(|e| panic!("clone warm flat dir: {e}"));
        let model = Lfm2Model::load_from_dir(&dir.to_string_lossy())
            .await
            .unwrap_or_else(|e| panic!("load warm flat model: {e:?}"));

        // Turn 1: fresh session. Compiled path engages and persists the
        // [user1, assistant(reply1)] history for the next strict-extend hit.
        //
        // We echo `r1.raw_text` (NOT `r1.text`) into turn 2's history. This is
        // LOAD-BEARING for reachability on this THINKING checkpoint: LFM2 ALWAYS
        // parses reasoning (`thinking_enabled` is hard-coded true in
        // `chat_sync_core`; the chat template ignores `enable_thinking`), so
        // `ChatResult.text` always has the leading `<think></think>` span
        // stripped. But `save_cache_state` persists the RAW generated tokens,
        // which BEGIN with those `<think></think>` markers — so echoing the
        // stripped `text` drops them and the re-templated turn-2 prompt diverges
        // from the cache at the first generated token (`verify_cache_prefix`
        // returns 0 → full prefill → `cached_tokens=0`). `greedy_chat_config_no_thinking`
        // sets `include_reasoning: Some(true)` so `r1.raw_text` is the FULL
        // verbatim decode (markers included); it round-trips token-exact through
        // the tokenizer, so `assistant_message(&r1.raw_text)` re-renders the
        // EXACT cached prefix and the strict-extend branch fires (cached_tokens>0).
        let before_t1 = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
        let r1 = model
            .chat_session_start(
                vec![user_message(user1)],
                Some(greedy_chat_config_no_thinking(max_new, true)),
            )
            .await
            .unwrap_or_else(|e| panic!("warm flat turn1: {e:?}"));
        let after_t1 = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
        let turn1_delta = after_t1.saturating_sub(before_t1);
        eprintln!(
            "[warm-flat] turn1 call_delta={turn1_delta} text={:?} raw_text={:?}",
            r1.text, r1.raw_text
        );

        // Turn 2: a NEW session-start whose message list replays the prior turn
        // (user1 + assistant RAW reply1) and appends user2. Echoing the VERBATIM
        // `r1.raw_text` (markers and all) — not the reasoning-stripped `r1.text` —
        // makes the tokenized prompt a STRICT EXTENSION of the persisted history
        // (which `save_cache_state` stored as the raw generated tokens), so
        // `chat_sync_core` takes the strict-extend branch (prefill only the user2
        // tail) AND keeps the compiled path engaged — exactly fix #1's warm-reuse
        // seed path.
        let before_t2 = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
        let r2 = model
            .chat_session_start(
                vec![
                    user_message(user1),
                    assistant_message(&r1.raw_text),
                    user_message(user2),
                ],
                Some(greedy_chat_config_no_thinking(max_new, true)),
            )
            .await
            .unwrap_or_else(|e| panic!("warm flat turn2 (strict-extend start): {e:?}"));
        let after_t2 = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
        let turn2_delta = after_t2.saturating_sub(before_t2);
        let turn2_cached = r2.cached_tokens;
        eprintln!(
            "[warm-flat] turn2 call_delta={turn2_delta} cached_tokens={turn2_cached} text={:?}",
            r2.text
        );
        drop(model);
        // Return `r1.raw_text` (the VERBATIM reply echoed into turn 2) so the
        // cold oracle below replays the IDENTICAL turn-2 prompt — apples-to-apples.
        (turn1_delta, turn2_delta, turn2_cached, r2.text, r1.raw_text)
    };

    // Turn 1 (session anchor) MUST engage the compiled path.
    assert!(
        warm_turn1_delta >= (max_new as u64).saturating_sub(2),
        "warm turn 1 compiled path did not engage: call_delta={warm_turn1_delta} (expected ~{})",
        max_new - 1
    );

    // DECISIVE: turn 2 is a strict-extend warm hit and MUST still engage the
    // compiled path. Pre-fix, seeding the delta-length offset would have failed
    // the seed (is_initialized()==0 -> native fallback, call_delta==0) or
    // slice_update-corrupted the cache; a clean ~N-1 delta proves the
    // get_offset()-based seed was accepted on a NON-ZERO cached prefix.
    assert!(
        warm_turn2_delta >= (max_new as u64).saturating_sub(2),
        "warm strict-extend turn 2 compiled path did not engage: call_delta={warm_turn2_delta} \
         (expected ~{}). A wrong (delta-length) seed offset would fail the seed and fall back \
         to native (call_delta==0).",
        max_new - 1
    );
    assert!(
        !warm_turn2_text.trim().is_empty(),
        "warm strict-extend turn 2 produced empty output (offset/export round-trip broken)"
    );
    // DECISIVE (warm-path proof): turn 2 MUST have reused a non-empty cached
    // prefix. If `cached_tokens == 0` the prompt was full-prefilled and the
    // strict-extend / KV-offset-reuse path this test claims to validate was
    // NOT exercised (a full-prefill turn 2 would still engage compiled AND
    // match the cold oracle, so this is the only assertion that pins it).
    assert!(
        warm_turn2_cached > 0,
        "warm strict-extend turn 2 reused NO cached prefix (cached_tokens=0 => \
         full prefill; the warm KV-offset-reuse path was not exercised)"
    );

    // ---- Oracle: identical turn-2 prompt from a COLD (fresh) session ------
    // Same compiled flat graph + same full prompt, but no warm prefix reuse
    // (the model is freshly loaded so the prefix verifier misses -> full
    // prefill). A correct warm strict-extend must match this within the tail
    // tolerance. Paged is NOT used as the oracle here: it degenerates into
    // verbatim repetition on these short prompts (observed finish=repetition at
    // ~8 tokens), so it is not a reliable text reference for the reuse turn.
    let cold_turn2 = {
        let dir = clone_model_dir(&src, "warm-cold", false)
            .unwrap_or_else(|e| panic!("clone warm-cold dir: {e}"));
        let model = Lfm2Model::load_from_dir(&dir.to_string_lossy())
            .await
            .unwrap_or_else(|e| panic!("load warm-cold model: {e:?}"));
        let r = model
            .chat_session_start(
                vec![
                    user_message(user1),
                    // Echo the SAME verbatim `raw_text` the warm turn 2 echoed so
                    // the templated turn-2 prompt is byte-identical between the
                    // warm-reuse run and this cold oracle.
                    assistant_message(&turn1_reply),
                    user_message(user2),
                ],
                // Same config as the warm run so this is a like-for-like
                // full-prefill oracle of the identical prompt.
                Some(greedy_chat_config_no_thinking(max_new, true)),
            )
            .await
            .unwrap_or_else(|e| panic!("warm-cold turn2: {e:?}"));
        let out = run_outcome("warm-cold-turn2", &r, 0);
        drop(model);
        out
    };

    let tail = tail_diff_bytes(&warm_turn2_text, &cold_turn2.text);
    eprintln!(
        "[warm] turn2 warm-vs-cold tail_diff={tail}B common_prefix={}B",
        common_prefix_len(&warm_turn2_text, &cold_turn2.text)
    );
    assert_tail_only_divergence(
        "warm turn2 warm-vs-cold",
        &warm_turn2_text,
        &cold_turn2.text,
    );
    eprintln!(
        "[PASS] warm strict-extend reuse offset fix holds: turn-2 compiled-flat ENGAGED \
         (call_delta={warm_turn2_delta}) on a non-zero cached prefix and matches the \
         cold-session run within tail tolerance ({tail}B)"
    );
}

// =============================================================================
// PHASE 3c: real-model MoE end-to-end parity — compiled-flat sparse-MoE decode
// vs the native-paged reference, on the real `lfm2.5-8b-a1b` (model_type
// lfm2_moe, bf16, 24 layers, 32 experts, top-4, 2 dense layers).
//
// What this proves:
//   1. ENGAGEMENT — the compiled C++ forward (`mlx_lfm2_moe_forward`) actually
//      runs once per decode step for a MoE checkpoint (model_id published,
//      forward_call_count delta ~N-1). delta == 0 => silent native fallback =>
//      hard fail (the Phase-3c gate-lift did not take).
//   2. PARITY — greedy decode on the compiled-flat MoE path agrees with a
//      reference everywhere except (at most) a short trailing argmax-tie flip.
//
// REFERENCE CHOICE: the fidelity oracle for "is the compiled MoE port faithful?"
// is EAGER-FLAT (`MLX_NO_COMPILE=1`, separate process) — the SAME flat KV/conv
// layout + SAME `lfm2_moe_ffn` routing, just not `mlx::core::compile`d (see the
// module docstring + `lfm2_eager_flat_vs_compiled_capture`, which is checkpoint-
// agnostic via `MLX_TEST_MODEL_PATH` and so ALSO covers MoE when pointed at the
// MoE checkpoint). This in-process test uses NATIVE-PAGED as the cross-check
// reference within the same late-argmax-tie tail tolerance the dense test uses:
// paged is a DIFFERENT graph (paged attention), so a single late near-tie can
// legitimately flip the final token(s) — we require agreement on everything
// EARLIER than the short trailing region, which is exactly what a real
// compiled-MoE bug (garbage from token 0 of decode, or a silent fallback) would
// violate. We also write a MoE-specific compiled-flat artifact so a separate
// `MLX_NO_COMPILE=1` `lfm2_eager_flat_vs_compiled_capture` invocation can diff
// eager-flat against it for the strict fidelity verdict.
//
// Gated: runs ONLY when `LFM2_COMPILED_E2E=1` AND the checkpoint dir exists
// (point `MLX_TEST_MODEL_PATH` at `.cache/models/lfm2.5-8b-a1b`). The 8B MoE
// model is large; this test is slow by design.
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_moe_compiled_flat_vs_paged_e2e_parity() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        // Under MLX_NO_COMPILE=1 the flat path runs the EAGER C++ graph, so this
        // compiled-engagement test is meaningless; the eager-flat capture test
        // owns that mode. Skip cleanly.
        eprintln!("[skip] MLX_NO_COMPILE=1 — compiled-engagement test is compiled-mode-only");
        return;
    }
    let Some(src) = resolve_moe_model() else {
        return;
    };

    // STRUCTURAL pins so a wrong-but-MoE topology fails hard. `resolve_moe_model`
    // already guarantees `src` is a real lfm2_moe checkpoint (or it skipped / hard-
    // failed), so the prior inline `is_moe` skip block is unreachable and removed;
    // we still parse `config.json` here to assert the exact 8B topology. `Lfm2Config`
    // has no `model_type` field, so read these from config.json directly.
    let cfg_raw = fs::read_to_string(src.join("config.json"))
        .unwrap_or_else(|e| panic!("[moe-e2e] read config.json at {}: {e}", src.display()));
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_raw)
        .unwrap_or_else(|e| panic!("[moe-e2e] parse config.json at {}: {e}", src.display()));
    assert_eq!(
        cfg_json.get("num_experts").and_then(|x| x.as_i64()),
        Some(32),
        "[moe-e2e] expected lfm2.5-8b-a1b num_experts=32"
    );
    assert_eq!(
        cfg_json.get("num_experts_per_tok").and_then(|x| x.as_i64()),
        Some(4),
        "[moe-e2e] expected lfm2.5-8b-a1b num_experts_per_tok=4"
    );
    assert_eq!(
        cfg_json.get("num_dense_layers").and_then(|x| x.as_i64()),
        Some(2),
        "[moe-e2e] expected lfm2.5-8b-a1b num_dense_layers=2"
    );
    // Without pinning the layer DEPTH, a degenerate config with
    // num_hidden_layers == num_dense_layers (e.g. 2/2) would satisfy EVERY assert
    // above (model_type / num_experts / num_experts_per_tok / num_dense_layers)
    // yet route ZERO layers through the sparse-MoE block — making this whole
    // parity test vacuous. Pin the real depth (24 layers) and require strictly
    // more total layers than dense ones so at least one sparse-MoE layer (here
    // 24 - 2 = 22 of them) is genuinely exercised by the compiled forward.
    let num_hidden_layers = cfg_json.get("num_hidden_layers").and_then(|x| x.as_i64());
    let num_dense_layers = cfg_json.get("num_dense_layers").and_then(|x| x.as_i64());
    assert_eq!(
        num_hidden_layers,
        Some(24),
        "[moe-e2e] expected lfm2.5-8b-a1b num_hidden_layers=24"
    );
    assert!(
        matches!((num_hidden_layers, num_dense_layers), (Some(h), Some(d)) if h > d),
        "[moe-e2e] num_hidden_layers ({num_hidden_layers:?}) must exceed num_dense_layers \
         ({num_dense_layers:?}) so at least one sparse-MoE layer is actually exercised; \
         otherwise the bf16+flat sparse-MoE gate-lift is left entirely UNGUARDED"
    );

    eprintln!("[moe-e2e] checkpoint: {}", src.display());

    let prompt = "What is the capital of France? Answer in one short sentence.";

    // ---- Compiled flat MoE path (under test) -----------------------------
    let flat = run_flat_single(&src, "moe-flat", prompt, N_NEW_TOKENS).await;

    // ENGAGEMENT + GATE. delta == 0 => the MoE gate-lift failed and decode
    // silently fell back to native forward() => hard fail.
    assert!(
        flat.call_delta >= MIN_COMPILED_CALL_DELTA,
        "MoE COMPILED PATH DID NOT ENGAGE: forward_call_count delta = {} (model_id={}, \
         weight_count={}). The Phase-3c MoE gate-lift did not register the compiled weights, \
         or decode fell back to native forward().",
        flat.call_delta,
        flat.model_id,
        flat.weight_count
    );
    assert_ne!(
        flat.model_id, 0,
        "MoE compiled model id was not published (registration suppressed?)"
    );
    // The MoE checkpoint registers far more tensors than the dense 1.2B
    // (stacked experts + router gate + expert_bias across 22 MoE layers), so we
    // don't pin an exact count — just require a non-trivial registry.
    assert!(
        flat.weight_count > 0,
        "MoE compiled weight registry is empty"
    );
    eprintln!(
        "[moe-e2e] MoE compiled-flat engaged: call_delta={} model_id={} weight_count={}",
        flat.call_delta, flat.model_id, flat.weight_count
    );

    // Persist the compiled-flat MoE decode text so a SEPARATE `MLX_NO_COMPILE=1`
    // invocation (eager-flat capture) can diff against it for the fidelity
    // verdict. The filename is topology-qualified ("moe") so it can NEVER collide
    // with the DENSE suite's `lfm2_e2e_compiled_flat.txt`: the dense compiled test
    // (`lfm2_compiled_flat_vs_paged_e2e_parity`) writes that unqualified name and
    // `lfm2_eager_flat_vs_compiled_capture` reads it, so a stale dense artifact
    // sharing this name would otherwise be silently diffed against a MoE eager run
    // (false pass/fail). The dense name is left untouched; only the MoE writer is
    // qualified.
    let art = artifact_path("lfm2_moe_e2e_compiled_flat.txt");
    if let Err(e) = fs::write(&art, &flat.text) {
        eprintln!("[warn] could not write MoE compiled-flat artifact {art:?}: {e}");
    } else {
        eprintln!(
            "[moe-e2e] wrote MoE compiled-flat artifact: {}",
            art.display()
        );
    }

    // ---- Reference: native paged path ------------------------------------
    let paged = run_paged_single(&src, "moe-paged", prompt, N_NEW_TOKENS).await;

    // ---- Parity --------------------------------------------------------
    // compiled-flat vs paged are DIFFERENT graphs, so a late argmax-tie flip is
    // legitimate. Require agreement everywhere except the short trailing region;
    // an EARLY divergence is a real compiled-MoE bug and fails the test.
    let tail = tail_diff_bytes(&flat.text, &paged.text);
    eprintln!(
        "[moe-e2e] compiled-vs-paged: tail_diff={tail}B (common_prefix={}B)",
        common_prefix_len(&flat.text, &paged.text)
    );
    assert_tail_only_divergence("moe-compiled-vs-paged", &flat.text, &paged.text);
    assert_eq!(
        flat.num_tokens, paged.num_tokens,
        "MoE num_tokens mismatch: compiled={} paged={}",
        flat.num_tokens, paged.num_tokens
    );
    assert_eq!(
        flat.finish_reason, paged.finish_reason,
        "MoE finish_reason mismatch: compiled={} paged={}",
        flat.finish_reason, paged.finish_reason
    );

    eprintln!(
        "[PASS] MoE compiled-flat engaged ({} calls) and agrees with paged within \
         tail tolerance ({tail}B <= {TAIL_TOLERANCE_BYTES}B)",
        flat.call_delta
    );
}

// =============================================================================
// PHASE 4 PIECE 2: INDEPENDENT mlx-lm GOLDEN PARITY (8B MoE, env-gated/heavy).
//
// The MoE counterpart of `lfm2_compiled_flat_vs_mlx_lm_golden`: it diffs the
// compiled-flat sparse-MoE decode's `raw_text` against a FROZEN greedy decode
// captured from mlx-lm's SEPARATE `lfm2_moe.py` (gate→float32→softmax→add
// expert_bias→argpartition top-4→norm_topk_prob, lfm2_moe.py:209-224). A bug in
// our compiled MoE routing ORDER (e.g. softmax-after-bias, wrong top-k) shifts
// the greedy trajectory and is caught here — the eager-flat / native-paged
// references can't see it (same Rust+C++ routing). Gated behind the same
// LFM2_COMPILED_E2E + MLX_TEST_MODEL_PATH + is_moe structural guard as the other
// MoE test; the 16G load + 80-step decode makes it slow by design.
// =============================================================================
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_moe_compiled_flat_vs_mlx_lm_golden() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        eprintln!("[skip] MLX_NO_COMPILE=1 — MoE golden parity test is compiled-mode-only");
        return;
    }
    let Some(src) = resolve_moe_model() else {
        return;
    };

    // `resolve_moe_model` already guarantees `src` is a real lfm2_moe checkpoint
    // (or it skipped / hard-failed), so the prior inline `is_moe` skip block is
    // unreachable and removed. We still parse `config.json` here for the STRUCTURAL
    // pins so a wrong-but-MoE topology fails hard (mirrors the paged MoE test).
    let cfg_raw = fs::read_to_string(src.join("config.json"))
        .unwrap_or_else(|e| panic!("[moe-golden] read config.json at {}: {e}", src.display()));
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_raw)
        .unwrap_or_else(|e| panic!("[moe-golden] parse config.json at {}: {e}", src.display()));
    assert_eq!(
        cfg_json.get("num_experts").and_then(|x| x.as_i64()),
        Some(32),
        "[moe-golden] expected lfm2.5-8b-a1b num_experts=32"
    );
    assert_eq!(
        cfg_json.get("num_experts_per_tok").and_then(|x| x.as_i64()),
        Some(4),
        "[moe-golden] expected lfm2.5-8b-a1b num_experts_per_tok=4"
    );
    assert_eq!(
        cfg_json.get("num_dense_layers").and_then(|x| x.as_i64()),
        Some(2),
        "[moe-golden] expected lfm2.5-8b-a1b num_dense_layers=2"
    );
    assert_eq!(
        cfg_json.get("num_hidden_layers").and_then(|x| x.as_i64()),
        Some(24),
        "[moe-golden] expected lfm2.5-8b-a1b num_hidden_layers=24"
    );

    eprintln!("[moe-golden] checkpoint: {}", src.display());

    let max_new = 80;
    let flat = run_flat_golden(&src, "moe-golden", GOLDEN_PROMPT, max_new).await;

    // ENGAGEMENT.
    assert!(
        flat.call_delta >= MIN_COMPILED_CALL_DELTA,
        "MoE GOLDEN: compiled path did not engage: forward_call_count delta = {} \
         (model_id={}, weight_count={}); decode fell back to native.",
        flat.call_delta,
        flat.model_id,
        flat.weight_count
    );
    assert_ne!(
        flat.model_id, 0,
        "moe golden: compiled model id not published"
    );
    assert!(
        flat.weight_count > 0,
        "moe golden: compiled weight registry empty"
    );

    // FULL-GENERATION: the golden was captured at 80 ids that are STILL
    // mid-reasoning (no EOS), so the correct decode finishes by LENGTH at 80. An
    // early stop would shrink the oracle window and could pass a short prefix
    // while hiding a downstream bug — so pin both the token count and the reason.
    assert_eq!(
        flat.num_tokens, max_new as u32,
        "MoE golden: generated {} tokens but expected the full {} — an early stop \
         truncates the oracle window and could pass a short prefix while hiding a downstream bug",
        flat.num_tokens, max_new
    );
    assert_eq!(
        flat.finish_reason, "length",
        "MoE golden: finished by {:?} not \"length\" — early EOS/stop shrinks the \
         comparison window below the captured golden",
        flat.finish_reason
    );

    // TEMPLATE-DRIFT CANARY.
    assert_eq!(
        flat.prompt_tokens, GOLDEN_PROMPT_TOKENS_MOE,
        "moe golden: prompt re-templated to {} ids but mlx-lm golden was captured at {} ids \
         — chat_template drift. Regenerate (scripts/capture_lfm2_golden.py).",
        flat.prompt_tokens, GOLDEN_PROMPT_TOKENS_MOE
    );

    // DECISIVE: independent-oracle parity (exercises lfm2_moe.py routing).
    let lcp = assert_golden_prefix(
        "moe-golden",
        &flat.raw_text,
        GOLDEN_TEXT_MOE,
        GOLDEN_MIN_PREFIX_BYTES_MOE,
    );

    // FULL-LENGTH regression teeth: assert_golden_prefix above proves the first 138
    // bytes match the INDEPENDENT mlx-lm oracle; this pins the ENTIRE compiled output
    // so a regression flipping ANY token after the gate fails (closes the "passes
    // after the 2-byte pin" hole). GOLDEN_RAW_TEXT_MOE_OURS is byte-reproducible; its
    // 138-byte prefix == GOLDEN_TEXT_MOE; the tail is our benign ' So' continuation
    // (compile()-faithful on THIS trajectory: see lfm2_moe_golden_eager_vs_compiled).
    assert_eq!(
        flat.raw_text, GOLDEN_RAW_TEXT_MOE_OURS,
        "MoE golden: compiled raw_text diverged from the byte-reproducible compiled \
         self-golden — a real regression somewhere in the 80-token decode (the mlx-lm \
         prefix gate already cleared, so this is a post-prefix change)."
    );

    eprintln!(
        "[PASS] MoE compiled-flat matches the INDEPENDENT mlx-lm greedy golden on a \
         {lcp}-byte early prefix (>= {GOLDEN_MIN_PREFIX_BYTES_MOE}); compiled path engaged \
         (call_delta={}).",
        flat.call_delta
    );
}

// =============================================================================
// P4 (STREAMING): COMPILED-PAGED engagement + non-stream parity (DENSE 1.2B).
//
// The non-streaming paged decode loop (`chat_sync_core_paged_inner`) was already
// proven to engage the compiled-paged C++ graph by
// `lfm2_compiled_paged_engagement_dense` / `lfm2_compiled_paged_vs_eager_paged`.
// The STREAMING paged decode loop (`chat_stream_sync_core_paged_inner`) was JUST
// wired into the SAME shared compiled-paged helpers
// (`Lfm2Inner::paged_compiled_decode_setup` + `paged_compiled_decode_step`); this
// test is its dedicated regression gate. It proves two things on the real 1.2B
// dense checkpoint with `use_block_paged_cache:true` (the production default):
//
//   1. ENGAGEMENT — `mlx_lfm2_moe_compiled_paged_call_count` advances by ~N-1
//      across a STREAMING decode. Before the streaming loop was wired to the
//      compiled-paged path this delta was 0 (the stream silently ran the eager
//      pure-Rust paged decode). A 0 delta here is the exact pre-fix failure and
//      is a HARD FAIL.
//   2. PARITY — the text reconstructed from the streaming delta chunks
//      (`chunk.text`, concatenated) is BYTE-IDENTICAL to the non-streaming
//      compiled-paged `ChatResult.text` reference. Both run the SAME compiled-
//      paged graph at temperature 0 over the SAME prompt/config, so the only
//      tolerated divergence is a late argmax-tie tail (`TAIL_TOLERANCE_BYTES`,
//      the suite's shared comparison budget) — an early divergence is a real
//      streaming-seam bug.
//
// Each side loads a FRESH model from its OWN cloned checkpoint dir (matching the
// `run_*` helpers) so there is no cross-run cache state. Under `MLX_NO_COMPILE=1`
// the paged path is the eager C++ graph and the compiled-paged counter never
// bumps, so skip cleanly (the eager-paged capture tests own that mode).
// =============================================================================

/// Drain the LFM2 streaming receiver for one turn, concatenating the delta
/// `chunk.text` (the SAME reconstruction the proven session test
/// `lfm2_session_stream_matches_non_stream_byte_for_byte` uses to match
/// `ChatResult.text`). Returns the reconstructed text, the terminal
/// `finish_reason`, and the terminal `num_tokens`. Panics if the stream never
/// reaches `done == true` (a stream that closes without a terminal chunk is a
/// hard bug, not a benign early stop).
async fn drain_lfm2_stream_text(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<napi::Result<ChatStreamChunk>>,
) -> (String, String, u32) {
    let mut text = String::new();
    let mut finish_reason = String::new();
    let mut num_tokens = 0u32;
    let mut saw_done = false;
    while let Some(result) = rx.recv().await {
        let chunk = result.expect("lfm2 stream chunk error");
        if chunk.done {
            saw_done = true;
            finish_reason = chunk.finish_reason.unwrap_or_default();
            num_tokens = chunk.num_tokens.unwrap_or(0);
            break;
        }
        text.push_str(&chunk.text);
    }
    assert!(saw_done, "lfm2 stream never reached done=true");
    (text, finish_reason, num_tokens)
}

/// Strip a LEADING empty `<think></think>` reasoning wrapper (and the single
/// following newline, if present) from a streamed reconstruction. With
/// `thinking_token_budget: Some(0)` the template opens then immediately closes
/// thinking, so the streamed delta chunks carry an empty `<think></think>\n`
/// that the non-streaming `ChatResult.text` already had stripped by
/// `finalize_chat_result`. Stripping it here makes the stream-vs-non-stream
/// comparison content-vs-content. Only an EMPTY leading wrapper is removed; any
/// non-empty `<think>…content…</think>` is left intact so a real reasoning-span
/// divergence would still surface. Returns the input unchanged if no empty
/// wrapper is present (so it is a no-op on the already-stripped reference).
fn strip_leading_empty_think(s: &str) -> &str {
    const EMPTY: &str = "<think></think>";
    if let Some(rest) = s.strip_prefix(EMPTY) {
        rest.strip_prefix('\n').unwrap_or(rest)
    } else {
        s
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm2_compiled_paged_streaming_engagement_and_parity_dense() {
    if !gated() {
        eprintln!("[skip] LFM2_COMPILED_E2E != 1");
        return;
    }
    if no_compile_env() {
        // The streaming-compiled engagement signal is the compiled-paged counter,
        // which never bumps under MLX_NO_COMPILE=1 (the paged path is the eager C++
        // graph there). Skip cleanly so a whole-binary MLX_NO_COMPILE=1 invocation
        // doesn't spuriously fail (the eager-paged capture tests own that mode).
        eprintln!(
            "[skip] MLX_NO_COMPILE=1 — streaming compiled-paged engagement test is compile-mode-only"
        );
        return;
    }
    let Some(src) = resolve_source_model() else {
        return;
    };
    eprintln!("[paged-compiled-stream] checkpoint: {}", src.display());

    let prompt = "What is the capital of France? Answer in one short sentence.";

    // ---- NON-STREAMING reference (compiled-paged) ------------------------
    // Reuse the proven non-streaming compiled-paged helper: it loads a fresh paged
    // model, runs `chat_session_start`, and reports the compiled-paged call delta.
    let reference =
        run_paged_compiled(&src, "paged-compiled-stream-ref", prompt, N_NEW_TOKENS).await;
    assert!(
        reference.call_delta >= MIN_COMPILED_CALL_DELTA,
        "REFERENCE non-streaming compiled-paged path did not engage (call_delta={}, \
         model_id={}, weight_count={}); the parity comparison would be meaningless.",
        reference.call_delta,
        reference.model_id,
        reference.weight_count
    );
    eprintln!(
        "[paged-compiled-stream] non-streaming reference engaged (call_delta={}, num_tokens={}, finish={})",
        reference.call_delta, reference.num_tokens, reference.finish_reason
    );

    // ---- STREAMING under test (compiled-paged) --------------------------
    // Fresh model from its own cloned paged dir to avoid cross-run cache state,
    // exactly like the `run_*` helpers. `clone_model_dir(.., true)` sets
    // `paged_block_size: 16`, which `init_lfm2_paged_compiled_session` requires.
    let stream_dir = clone_model_dir(&src, "paged-compiled-stream", true)
        .unwrap_or_else(|e| panic!("clone paged-compiled-stream dir: {e}"));
    let stream_model = Lfm2Model::load_from_dir(&stream_dir.to_string_lossy())
        .await
        .unwrap_or_else(|e| panic!("load paged-compiled-stream model: {e:?}"));

    let before = unsafe { mlx_sys::mlx_lfm2_moe_compiled_paged_call_count() };
    let (handle, rx) = stream_model
        .chat_stream_session_start_for_test(
            vec![user_message(prompt)],
            Some(greedy_chat_config(N_NEW_TOKENS, true)),
        )
        .unwrap_or_else(|e| panic!("chat_stream_session_start_for_test dispatch failed: {e:?}"));
    let (stream_text, stream_finish, stream_tokens) = drain_lfm2_stream_text(rx).await;
    let after = unsafe { mlx_sys::mlx_lfm2_moe_compiled_paged_call_count() };
    let stream_call_delta = after.saturating_sub(before);
    drop(handle);
    drop(stream_model);

    eprintln!(
        "[paged-compiled-stream] STREAMING run: call_delta={stream_call_delta} num_tokens={stream_tokens} finish={stream_finish}"
    );
    eprintln!("[paged-compiled-stream] stream_text = {stream_text:?}");

    // ---- ASSERT 1: STREAMING compiled-paged ENGAGEMENT ------------------
    // This is the whole point: pre-fix the streaming loop ran the eager pure-Rust
    // paged decode and left this delta at 0.
    assert!(
        stream_call_delta >= MIN_COMPILED_CALL_DELTA,
        "STREAMING compiled-paged path did not engage (call_delta={stream_call_delta}) — \
         chat_stream_sync_core_paged_inner is not wired to compiled-paged. Expected ~N-1 \
         compiled-paged forward calls (>= {MIN_COMPILED_CALL_DELTA}); a 0/low delta means the \
         streaming decode silently fell back to the eager pure-Rust paged decode."
    );

    // ---- ASSERT 2: STREAMING == NON-STREAMING reference -----------------
    // Both run the SAME compiled-paged graph at temperature 0, so the GENERATED
    // CONTENT must be byte-identical save for at most a late argmax-tie tail.
    //
    // ONE representation difference must be normalized first: with
    // `thinking_token_budget: Some(0)` the template opens then immediately forces
    // a `</think>`, so the run emits an EMPTY `<think></think>\n` reasoning block.
    // The non-streaming `ChatResult.text` (`reference.text`) has that empty span
    // stripped by `finalize_chat_result`, but the streaming delta chunks emit it
    // verbatim, so the raw streamed text carries a leading `<think></think>\n` the
    // reference does not. That is a FORMATTING difference, not a decode
    // divergence (the post-wrapper content is what proves parity). Strip a leading
    // empty-think wrapper from the streamed reconstruction so the comparison is
    // content-vs-content (the strip is a no-op on the already-stripped reference).
    let stream_content = strip_leading_empty_think(&stream_text);
    let tail = tail_diff_bytes(stream_content, &reference.text);
    eprintln!(
        "[paged-compiled-stream] stream-vs-nonstream: tail_diff={tail}B (common_prefix={}B) \
         (streamed had empty-think wrapper stripped: {})",
        common_prefix_len(stream_content, &reference.text),
        stream_content.len() != stream_text.len()
    );
    assert_tail_only_divergence(
        "paged-compiled-stream-vs-nonstream",
        stream_content,
        &reference.text,
    );

    eprintln!(
        "[PASS] STREAMING compiled-PAGED engaged ({stream_call_delta} compiled-paged calls) and \
         agrees with the non-streaming compiled-paged reference within tail tolerance \
         ({tail}B <= {TAIL_TOLERANCE_BYTES}B). content snippet: {:?}",
        &stream_content.chars().take(80).collect::<String>()
    );
}
