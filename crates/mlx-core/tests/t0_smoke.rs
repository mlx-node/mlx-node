//! T=0 byte-equivalence smoke harness for the chat-engine refactor (S0).
//!
//! One test per model family: qwen3, qwen3_5, qwen3_5_moe, gemma4, lfm2.
//! Each test runs a deterministic 3-turn session (session_start, session_continue,
//! session_continue_tool) at temperature=0 and writes a canonical JSON digest
//! to `$MLX_SMOKE_OUT/<family>.json` when `MLX_SMOKE_OUT` is set, or prints
//! `SMOKE <family> turn=N field=value` lines to stdout otherwise.
//!
//! The digests are byte-comparable across runs so the `compare` mode of
//! `scripts/t0-smoke.sh` can detect regressions by diffing two captures.
//!
//! All tests are gated on env vars — skipped silently when unset so plain
//! `cargo test` (CI) is unaffected.
//!
//! ## Digest fields (S0 review fix, Finding 2)
//!
//! Option (c) was chosen: no new model accessor needed. `include_reasoning`
//! is set to `true` so `raw_text` contains the full decoded output including
//! thinking tokens. We also capture `thinking`, `prompt_tokens`,
//! `reasoning_tokens`, and `tool_calls` from `ChatResult` — together these
//! cover every distinct token-stream outcome. Token identity is implied by
//! `raw_text` (T=0 → same token IDs ↔ same decoded bytes) and `num_tokens`.
//! No option (a)/(b) accessor was added because no existing public API
//! returns the raw generated token ID vector, and the existing parity tests
//! likewise compare `text`/`raw_text` rather than token IDs.
//!
//! Run a single family manually with e.g.:
//! ```shell
//! MLX_SMOKE_QWEN3_MODEL_PATH=/Volumes/P4510/models/qwen3-0.6b-mlx-bf16 \
//! MLX_SMOKE_OUT=/tmp/smoke-out \
//!   cargo test -p mlx-core --test t0_smoke qwen3_smoke -- --exact --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use mlx_core::models::qwen3_5::model::{ChatConfig, ChatResult};
use mlx_core::tokenizer::ChatMessage;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn smoke_chat_config() -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(64),
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
        // S0 review fix (Finding 2): include_reasoning=true so raw_text carries
        // the full decoded output including thinking tokens. Together with the
        // thinking/prompt_tokens/reasoning_tokens fields in result_digest this
        // ensures that any token-stream divergence will appear in the digest.
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(true),
        enable_mtp: Some(false),
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

/// Collect the stable fields from a ChatResult into a sorted BTreeMap.
///
/// S0 review fix (Finding 2, option c): added thinking, prompt_tokens,
/// reasoning_tokens, tool_calls. include_reasoning is now Some(true) so
/// raw_text carries the full decoded output (reasoning included). Any
/// token-stream divergence surfaces in raw_text (T=0 ↔ same IDs ↔ same bytes)
/// and the extra fields catch reasoning/tool divergence independently.
fn result_digest(r: &ChatResult) -> BTreeMap<&'static str, String> {
    let mut m = BTreeMap::new();
    m.insert("cached_tokens", r.cached_tokens.to_string());
    m.insert("finish_reason", r.finish_reason.clone());
    m.insert("num_tokens", r.num_tokens.to_string());
    m.insert("prompt_tokens", r.prompt_tokens.to_string());
    m.insert("reasoning_tokens", r.reasoning_tokens.to_string());
    // raw_text: full decoded output (includes thinking tokens when
    // include_reasoning=true). Primary token-identity signal.
    m.insert("raw_text", r.raw_text.clone());
    m.insert("text", r.text.clone());
    m.insert("thinking", r.thinking.clone().unwrap_or_default());
    // Render tool_calls as a stable JSON string (BTreeMap keys → sorted).
    let tool_calls_json = serde_json::to_string(&r.tool_calls).unwrap_or_default();
    m.insert("tool_calls", tool_calls_json);
    m
}

/// Emit digest as `SMOKE <family> turn=N key=value` lines (stdout) OR write
/// a JSON file to `$MLX_SMOKE_OUT/<family>.json`.
///
/// The JSON format is a sorted object of turn objects, each a sorted object
/// of fields — byte-stable across runs.
fn emit_digest(family: &str, turns: &[(usize, BTreeMap<&'static str, String>)]) {
    let out_dir = std::env::var("MLX_SMOKE_OUT").ok();
    if let Some(dir) = &out_dir {
        // Build canonical JSON: outer keys "turn_1".."turn_N", inner keys sorted.
        let mut outer = BTreeMap::new();
        for (turn_num, fields) in turns {
            let mut inner = BTreeMap::new();
            for (k, v) in fields {
                inner.insert(*k, v.clone());
            }
            outer.insert(format!("turn_{turn_num}"), inner);
        }
        let json = serde_json::to_string_pretty(&outer).expect("json serialization failed");
        std::fs::create_dir_all(dir).expect("create MLX_SMOKE_OUT dir");
        let path = format!("{dir}/{family}.json");
        std::fs::write(&path, json).expect("write digest file");
        eprintln!("SMOKE {family}: wrote digest to {path}");
    } else {
        for (turn_num, fields) in turns {
            for (k, v) in fields {
                println!("SMOKE {family} turn={turn_num} {k}={v}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// qwen3
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_SMOKE_QWEN3_MODEL_PATH pointing to a real Qwen3 checkpoint"]
async fn qwen3_smoke() {
    let Ok(model_path) = std::env::var("MLX_SMOKE_QWEN3_MODEL_PATH") else {
        eprintln!(
            "skipping qwen3_smoke: MLX_SMOKE_QWEN3_MODEL_PATH unset \
             (e.g. /Volumes/P4510/models/qwen3-0.6b-mlx-bf16)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("skipping qwen3_smoke: {model_path} does not exist");
        return;
    }

    let model = mlx_core::models::qwen3::persistence::load_with_thread(&model_path)
        .await
        .expect("failed to load Qwen3 model");

    // Turn 1: session start
    let r1 = model
        .chat_session_start(
            vec![user_message("What is the capital of France? One word.")],
            Some(smoke_chat_config()),
        )
        .await
        .expect("qwen3 turn 1 chat_session_start failed");
    eprintln!("qwen3 turn1 text={:?}", r1.text);

    // Turn 2: session continue
    let r2 = model
        .chat_session_continue(
            "Name another European capital. One word.".to_string(),
            None,
            Some(smoke_chat_config()),
        )
        .await
        .expect("qwen3 turn 2 chat_session_continue failed");
    eprintln!("qwen3 turn2 text={:?}", r2.text);

    // Turn 3: tool-result turn
    let r3 = model
        .chat_session_continue_tool(
            "call_abc123".to_string(),
            "The result of the lookup is: Rome".to_string(),
            Some(smoke_chat_config()),
            None,
        )
        .await
        .expect("qwen3 turn 3 chat_session_continue_tool failed");
    eprintln!("qwen3 turn3 text={:?}", r3.text);

    let turns = vec![
        (1, result_digest(&r1)),
        (2, result_digest(&r2)),
        (3, result_digest(&r3)),
    ];
    emit_digest("qwen3", &turns);
}

// ---------------------------------------------------------------------------
// qwen3_5
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_SMOKE_QWEN35_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn qwen3_5_smoke() {
    let Ok(model_path) = std::env::var("MLX_SMOKE_QWEN35_MODEL_PATH") else {
        eprintln!(
            "skipping qwen3_5_smoke: MLX_SMOKE_QWEN35_MODEL_PATH unset \
             (e.g. /Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("skipping qwen3_5_smoke: {model_path} does not exist");
        return;
    }

    let model = mlx_core::models::qwen3_5::model::Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");

    // Turn 1: session start
    let r1 = model
        .chat_session_start(
            vec![user_message("What is the capital of France? One word.")],
            Some(smoke_chat_config()),
        )
        .await
        .expect("qwen3_5 turn 1 chat_session_start failed");
    eprintln!("qwen3_5 turn1 text={:?}", r1.text);

    // Turn 2: session continue
    let r2 = model
        .chat_session_continue(
            "Name another European capital. One word.".to_string(),
            None,
            Some(smoke_chat_config()),
        )
        .await
        .expect("qwen3_5 turn 2 chat_session_continue failed");
    eprintln!("qwen3_5 turn2 text={:?}", r2.text);

    // Turn 3: tool-result turn
    let r3 = model
        .chat_session_continue_tool(
            "call_abc123".to_string(),
            "The result of the lookup is: Rome".to_string(),
            Some(smoke_chat_config()),
            None,
        )
        .await
        .expect("qwen3_5 turn 3 chat_session_continue_tool failed");
    eprintln!("qwen3_5 turn3 text={:?}", r3.text);

    let turns = vec![
        (1, result_digest(&r1)),
        (2, result_digest(&r2)),
        (3, result_digest(&r3)),
    ];
    emit_digest("qwen3_5", &turns);
}

// ---------------------------------------------------------------------------
// qwen3_5_moe
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_SMOKE_QWEN35MOE_MODEL_PATH pointing to a real Qwen3.5 MoE checkpoint"]
async fn qwen3_5_moe_smoke() {
    let Ok(model_path) = std::env::var("MLX_SMOKE_QWEN35MOE_MODEL_PATH") else {
        eprintln!(
            "skipping qwen3_5_moe_smoke: MLX_SMOKE_QWEN35MOE_MODEL_PATH unset \
             (e.g. /Volumes/P4510/models/Qwen3.6-35b-a3b-UD-Q4_K_XL-mlx)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("skipping qwen3_5_moe_smoke: {model_path} does not exist");
        return;
    }

    let model = mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 MoE model");

    // Turn 1: session start
    let r1 = model
        .chat_session_start(
            vec![user_message("What is the capital of France? One word.")],
            Some(smoke_chat_config()),
        )
        .await
        .expect("qwen3_5_moe turn 1 chat_session_start failed");
    eprintln!("qwen3_5_moe turn1 text={:?}", r1.text);

    // Turn 2: session continue
    let r2 = model
        .chat_session_continue(
            "Name another European capital. One word.".to_string(),
            None,
            Some(smoke_chat_config()),
        )
        .await
        .expect("qwen3_5_moe turn 2 chat_session_continue failed");
    eprintln!("qwen3_5_moe turn2 text={:?}", r2.text);

    // Turn 3: tool-result turn
    let r3 = model
        .chat_session_continue_tool(
            "call_abc123".to_string(),
            "The result of the lookup is: Rome".to_string(),
            Some(smoke_chat_config()),
            None,
        )
        .await
        .expect("qwen3_5_moe turn 3 chat_session_continue_tool failed");
    eprintln!("qwen3_5_moe turn3 text={:?}", r3.text);

    let turns = vec![
        (1, result_digest(&r1)),
        (2, result_digest(&r2)),
        (3, result_digest(&r3)),
    ];
    emit_digest("qwen3_5_moe", &turns);
}

// ---------------------------------------------------------------------------
// gemma4
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_SMOKE_GEMMA4_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_smoke() {
    let Ok(model_path) = std::env::var("MLX_SMOKE_GEMMA4_MODEL_PATH") else {
        eprintln!(
            "skipping gemma4_smoke: MLX_SMOKE_GEMMA4_MODEL_PATH unset \
             (e.g. /Volumes/P4510/models/gemma-4-e2b-it-mlx)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("skipping gemma4_smoke: {model_path} does not exist");
        return;
    }

    let model = mlx_core::models::gemma4::model::Gemma4Model::load(model_path.clone())
        .await
        .expect("failed to load Gemma4 model");

    // Turn 1: session start
    let r1 = model
        .chat_session_start(
            vec![user_message("What is the capital of France? One word.")],
            Some(smoke_chat_config()),
        )
        .await
        .expect("gemma4 turn 1 chat_session_start failed");
    eprintln!("gemma4 turn1 text={:?}", r1.text);

    // Turn 2: session continue
    let r2 = model
        .chat_session_continue(
            "Name another European capital. One word.".to_string(),
            None,
            Some(smoke_chat_config()),
        )
        .await
        .expect("gemma4 turn 2 chat_session_continue failed");
    eprintln!("gemma4 turn2 text={:?}", r2.text);

    // Turn 3: tool-result turn
    let r3 = model
        .chat_session_continue_tool(
            "call_abc123".to_string(),
            "The result of the lookup is: Rome".to_string(),
            Some(smoke_chat_config()),
            None,
        )
        .await
        .expect("gemma4 turn 3 chat_session_continue_tool failed");
    eprintln!("gemma4 turn3 text={:?}", r3.text);

    let turns = vec![
        (1, result_digest(&r1)),
        (2, result_digest(&r2)),
        (3, result_digest(&r3)),
    ];
    emit_digest("gemma4", &turns);
}

// ---------------------------------------------------------------------------
// lfm2
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_SMOKE_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_smoke() {
    let Ok(model_path) = std::env::var("MLX_SMOKE_LFM2_MODEL_PATH") else {
        eprintln!(
            "skipping lfm2_smoke: MLX_SMOKE_LFM2_MODEL_PATH unset \
             (e.g. /Volumes/P4510/models/lfm2.5-1.2b-thinking-mlx)"
        );
        return;
    };
    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("skipping lfm2_smoke: {model_path} does not exist");
        return;
    }

    let model = mlx_core::models::lfm2::model::Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // Turn 1: session start
    let r1 = model
        .chat_session_start(
            vec![user_message("What is the capital of France? One word.")],
            Some(smoke_chat_config()),
        )
        .await
        .expect("lfm2 turn 1 chat_session_start failed");
    eprintln!("lfm2 turn1 text={:?}", r1.text);

    // Turn 2: session continue
    let r2 = model
        .chat_session_continue(
            "Name another European capital. One word.".to_string(),
            None,
            Some(smoke_chat_config()),
        )
        .await
        .expect("lfm2 turn 2 chat_session_continue failed");
    eprintln!("lfm2 turn2 text={:?}", r2.text);

    // Turn 3: tool-result turn
    let r3 = model
        .chat_session_continue_tool(
            "call_abc123".to_string(),
            "The result of the lookup is: Rome".to_string(),
            Some(smoke_chat_config()),
            None,
        )
        .await
        .expect("lfm2 turn 3 chat_session_continue_tool failed");
    eprintln!("lfm2 turn3 text={:?}", r3.text);

    let turns = vec![
        (1, result_digest(&r1)),
        (2, result_digest(&r2)),
        (3, result_digest(&r3)),
    ];
    emit_digest("lfm2", &turns);
}
