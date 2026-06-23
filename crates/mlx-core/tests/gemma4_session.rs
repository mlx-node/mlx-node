//! Gated integration tests for Gemma4's native prefix-KV-cache reuse
//! path on `chat_session_start`.
//!
//! Covers the stateless-agent pattern (pi-mono / Aider / Codex clients
//! resend the full conversation every turn, never using
//! `previous_response_id`). See
//! `.claude/plans/dapper-zooming-catmull.md` for the full design.
//!
//! The tests are gated on `MLX_TEST_GEMMA4_MODEL_PATH` because they need
//! a real Gemma4 checkpoint on disk. Run manually with:
//!
//! ```shell
//! MLX_TEST_GEMMA4_MODEL_PATH=./.cache/models/gemma4-e2b-it-mlx-bf16 \
//!     cargo test -p mlx-core --test gemma4_session -- --ignored --nocapture
//! ```

use std::path::Path;

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::gemma4::model::Gemma4Model;
use mlx_core::tokenizer::ChatMessage;

fn chat_config_default(max_new_tokens: i32) -> ChatConfig {
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
        thinking_token_budget: None,
        include_reasoning: Some(false),
        report_performance: Some(true),
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
        audio: None,
    }
}

/// Append hit: turn 2's `chat_session_start` prompt is a strict
/// extension of turn 1's saved history. The reported `cached_tokens` on
/// turn 2 must be > 0 and the reply must still terminate cleanly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_session_start_prefix_reuse_append_hit() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_GEMMA4_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Gemma4Model::load(model_path.clone())
        .await
        .expect("failed to load Gemma4 model");

    // Turn 1: plain session start.
    let cfg1 = chat_config_default(32);
    let r1 = model
        .chat_session_start(vec![user_message("In exactly one short word, and nothing else, with no explanation and no punctuation and no preamble whatsoever, how would you warmly and politely greet a brand-new friend that you happen to be meeting for the very first time on this fine and bright sunny morning, bearing in mind that they have travelled a very long way over the hills and across the wide river and through the quiet forest to come and see you today and would dearly appreciate a simple kind and gentle word of welcome from you?")], Some(cfg1))
        .await
        .expect("turn 1 chat_session_start failed");
    assert_eq!(
        r1.cached_tokens, 0,
        "turn 1 should cold-start: cached_tokens={}",
        r1.cached_tokens
    );

    // Turn 2: resend the full transcript extended by a fresh user turn.
    // Gemma4's chat template renders "assistant" -> "model" turn blocks
    // deterministically, so the new prompt is a strict prefix extension
    // of the cached history.
    let cfg2 = chat_config_default(32);
    let turn2_msgs = vec![
        user_message(
            "In exactly one short word, and nothing else, with no explanation and no punctuation and no preamble whatsoever, how would you warmly and politely greet a brand-new friend that you happen to be meeting for the very first time on this fine and bright sunny morning, bearing in mind that they have travelled a very long way over the hills and across the wide river and through the quiet forest to come and see you today and would dearly appreciate a simple kind and gentle word of welcome from you?",
        ),
        ChatMessage {
            role: "assistant".to_string(),
            content: r1.text.clone(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            images: None,
            audio: None,
        },
        user_message("And another one?"),
    ];
    let r2 = model
        .chat_session_start(turn2_msgs, Some(cfg2))
        .await
        .expect("turn 2 chat_session_start failed");
    assert!(
        r2.cached_tokens > 0,
        "turn 2 expected a prefix cache hit but cached_tokens=0 \
         (turn1 tokens={}, turn2 tokens={})",
        r1.prompt_tokens,
        r2.prompt_tokens
    );
    assert!(
        r2.finish_reason == "stop" || r2.finish_reason == "length",
        "unexpected finish_reason on turn 2: {}",
        r2.finish_reason
    );
    // On a prefix-reuse hit the engine reprocesses only the new suffix: the
    // matched prefix (turn 1's rendered transcript) is served from the
    // content-addressed cache and `cached_tokens` reports it. Assert that
    // token accounting directly instead of wall-clock TTFT — TTFT is a flaky
    // CI signal (tiny warm-GPU times dominated by fixed per-turn setup, not
    // prefill work) AND a weak one (a broken reuse that silently re-prefilled
    // the whole prompt could still report a fast TTFT on a fast machine). The
    // deterministic proof that "only the delta was prefilled" is that the
    // freshly-prefilled span is smaller than the reused prefix.
    let uncached_delta = r2.prompt_tokens.saturating_sub(r2.cached_tokens);
    eprintln!(
        "prefix-reuse token accounting: prompt_tokens={} cached_tokens={} uncached_delta={}",
        r2.prompt_tokens, r2.cached_tokens, uncached_delta,
    );
    // The reused prefix must be the clear MAJORITY of the work: turn 2
    // reprocesses only the short new suffix (the one-word reply + the new
    // user turn), while turn 1's long question is served from the
    // content-addressed cache. `cached_tokens >= 2 * uncached_delta` proves
    // that deterministically — a broken reuse that re-prefilled the whole
    // prompt would push uncached_delta up to the full prompt and fail.
    assert!(
        uncached_delta * 2 < r2.cached_tokens,
        "prefix reuse is not the clear majority of the work: reused \
         prefix={} freshly-prefilled suffix={} (prompt_tokens={})",
        r2.cached_tokens,
        uncached_delta,
        r2.prompt_tokens,
    );
}

/// Divergence miss: turn 2's prompt begins with a completely different
/// first message than turn 1's cached history. The reported
/// `cached_tokens` on turn 2 must be 0 (full reset + full re-prefill)
/// and the reply must still be well-formed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_session_start_prefix_reuse_divergence_miss() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_GEMMA4_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Gemma4Model::load(model_path.clone())
        .await
        .expect("failed to load Gemma4 model");

    // Turn 1: prime the session with prompt A.
    let cfg1 = chat_config_default(32);
    let _ = model
        .chat_session_start(vec![user_message("Tell me a color.")], Some(cfg1))
        .await
        .expect("turn 1 chat_session_start failed");

    // Turn 2: start with an unrelated prompt B. This MUST miss and
    // reset; nothing from the turn-1 history is a prefix of this one.
    let cfg2 = chat_config_default(32);
    let r2 = model
        .chat_session_start(
            vec![user_message("Write a single short poem line.")],
            Some(cfg2),
        )
        .await
        .expect("turn 2 chat_session_start failed");

    assert_eq!(
        r2.cached_tokens, 0,
        "divergent-history turn 2 should miss but cached_tokens={}",
        r2.cached_tokens
    );
    assert!(
        r2.finish_reason == "stop" || r2.finish_reason == "length",
        "unexpected finish_reason on divergent turn 2: {}",
        r2.finish_reason
    );
    assert!(
        r2.num_tokens > 0,
        "divergent-history turn 2 produced no tokens: {:?}",
        r2
    );
}

/// Explicit `reset_caches` (`ResetScope::Command`) must purge the paged
/// adapter's content-addressed prefix cache, so a same-prompt rerun
/// replays a COLD full-prompt prefill (`cached_tokens == 0`) rather than
/// the prefix-hit 1-token-suffix prefill — whose different bf16 reduction
/// order can flip a greedy near-tie. gemma4 ships paged ON by default
/// (`use_block_paged_cache` defaults true), so the stock e2b checkpoint
/// already exercises the paged path with no config forcing. WITHOUT the
/// purge, turn 1's full blocks survive content-addressed and turn 2
/// reports `cached_tokens > 0` (deterministic failure); WITH the purge
/// they are gone and turn 2 cold-prefills (`cached_tokens == 0`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_session_reset_purges_prefix_cache_cold_prefill() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_GEMMA4_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Gemma4Model::load(model_path.clone())
        .await
        .expect("failed to load Gemma4 model");

    // The prompt MUST render to MORE than one paged block (block_size=16)
    // for this assertion to bite: the prefix lookup is capped at
    // `max_cache_hit_tokens = total_budget - 1 = prompt_len - 1`, and
    // `find_longest_cache_hit` only matches COMPLETE blocks, so a
    // single-block (<=16-token) prompt always reports `cached_tokens == 0`
    // on turn 2 regardless of the purge (the cap truncates the lookup
    // below the 16-token block boundary). This multi-sentence prompt is
    // comfortably >= 33 tokens (>= 2 full blocks), so WITHOUT the purge
    // turn 2 takes a >= 16-token prefix hit (`cached_tokens > 0`) and
    // WITH the purge cold-prefills (`cached_tokens == 0`).
    let prompt = "Please explain, in a few clear sentences, why the sky \
                  appears blue during the day and turns orange and red near \
                  sunset. Keep the explanation simple and friendly.";

    // Turn 1: cold session start. Primes the prefix cache with this
    // prompt's full blocks.
    let r1 = model
        .chat_session_start(vec![user_message(prompt)], Some(chat_config_default(32)))
        .await
        .expect("turn 1 chat_session_start failed");
    assert_eq!(
        r1.cached_tokens, 0,
        "turn 1 should cold-start: cached_tokens={}",
        r1.cached_tokens
    );

    // Explicit Command reset via the sync NAPI method. block_in_place:
    // reset_caches blocks on blocking_recv, which panics on a tokio
    // worker thread (see lfm2_session.rs + 8d5283a7 precedent).
    tokio::task::block_in_place(|| model.reset_caches()).expect("reset_caches failed");

    // Turn 2: rerun the IDENTICAL prompt as a fresh session start. The
    // only way cached_tokens could be > 0 is a surviving prefix entry —
    // exactly what the Command reset purges.
    let r2 = model
        .chat_session_start(vec![user_message(prompt)], Some(chat_config_default(32)))
        .await
        .expect("turn 2 chat_session_start after reset_caches failed");

    // PRIMARY deterministic gate (independent of any bf16 near-tie).
    assert_eq!(
        r2.cached_tokens, 0,
        "post-Command-reset same-prompt turn must cold-prefill (prefix cache \
         purged), but cached_tokens={} (turn1 prompt_tokens={}, turn2 \
         prompt_tokens={})",
        r2.cached_tokens, r1.prompt_tokens, r2.prompt_tokens
    );

    // SECONDARY byte-equality proof that the cold output is reproduced
    // (catches the actual greedy flip the divergence would cause).
    assert_eq!(
        r1.raw_text, r2.raw_text,
        "reset+rerun did not reproduce turn-1 raw_text byte-for-byte: \
         before={:?} after={:?}",
        r1.raw_text, r2.raw_text
    );
    assert_eq!(
        r1.text, r2.text,
        "reset+rerun did not reproduce turn-1 text byte-for-byte: \
         before={:?} after={:?}",
        r1.text, r2.text
    );
    assert_eq!(
        r1.num_tokens, r2.num_tokens,
        "reset+rerun did not reproduce turn-1 num_tokens: before={} after={}",
        r1.num_tokens, r2.num_tokens
    );
}
