//! Real-weights cold-tier restart-parity gates for Gemma4 (hybrid).
//!
//! Same three-instance scenario as the qwen3 gate — see
//! [`cold_tier_parity_harness`] — but gemma4 is the first HYBRID family to run
//! it, and that changes what a pass proves.
//!
//! # Why a pass here is sidecar evidence, not just KV evidence
//!
//! Gemma4 sizes its paged pool over the FULL-ATTENTION layers only; the
//! interleaved sliding layers keep their K/V in per-layer `RotatingKVCache`s
//! outside the pool. So its `ColdTierContext` carries a
//! `ColdSidecarPolicy`, and with a policy installed
//! `ColdTierWalk::restore_extend` restores NOTHING unless a validated sidecar
//! backs the boundary (`deepest_backed_boundary` -> `None` =>
//! `ColdRestore::miss()`).
//!
//! Instance 2 is a freshly loaded model, so its in-process prefix cache is
//! empty and a hot hit is impossible. Any non-zero `cached_tokens` it reports
//! can therefore only have come through the reconcile-down walk — which means
//! a sliding-window sidecar was found on disk, decoded, and validated against
//! this checkpoint's geometry.
//!
//! # Two gates, one per side of `min(boundary, window)`
//!
//! The sidecar payload carries `min(boundary, window)` token rows, which is
//! exactly what a live `RotatingKVCache` holds at that offset. The two regimes
//! either side of that `min` are structurally different states, so each gets
//! its own gate:
//!
//! | gate | boundary | rotating state restored |
//! |------|----------|-------------------------|
//! | [`gemma4_cold_tier_restart_parity`] | `>= window` | POST-wrap: a full window of rows, `idx` inside the ring |
//! | [`gemma4_cold_tier_restart_parity_sub_window`] | `< window` | PRE-wrap: `boundary` rows, `idx` derived from the offset |
//!
//! The sub-window gate is the one that would have been impossible before the
//! window floor came out of `boundary_is_representable`: gemma4's real window
//! is 1024 and a typical chat prompt is shorter, so under the old rule those
//! prompts had no representable boundary at all — nothing was ever backed,
//! while capture kept writing K/V blocks no restore could read back. It also
//! carries the stricter assertions, because a sub-window hit is precisely the
//! case where a "restore the K/V and replay the sliding half" fall-through
//! would look identical from the outside. See its own doc comment.
//!
//! # Why the long-prompt gate's prompt is sized the way it is
//!
//! Two opposing constraints pin it into a narrow band.
//!
//! **Floor.** Its whole job is the `>= window` regime, so the prompt must clear
//! one window — otherwise the only reachable boundary is a sub-window one and
//! the gate silently becomes a duplicate of its sibling. Its
//! `min_restored_tokens` floor of one whole `sliding_window` is what pins that
//! down; it is a statement about THIS fixture (its shallowest checkpoint is the
//! decode-cadence one at exactly one window), not a structural claim about the
//! layout.
//!
//! **Ceiling.** This fixture predates the cold anchor rungs and is still sized
//! for the world without them, deliberately: it keeps the prompt inside the
//! SECOND window, so the only decode-cadence checkpoint is the one at
//! `sliding_window`, which is both retained and the first boundary a growing
//! chain can cover. That makes it a gate on the `>= window` PAYLOAD regime and
//! nothing else — a prompt several windows long used to retain only its
//! DEEPEST two checkpoints (`gemma4_sliding_prefix_checkpoint_limit` works out
//! to TWO entries on a 26B) and evict the one at 1024, leaving nothing at or
//! below the chain's reach.
//!
//! That eviction is the bug `gemma4_sliding_cold_anchor_rungs` fixes, and it is
//! covered offline by
//! `gemma4_sliding_ladder_retains_a_rung_the_lagging_chain_can_reach` (which
//! replays the exact failing sequence) plus a real-weights `mlx agent` restart.
//! Sizing this gate up to lean on the rungs would make it a duplicate of that
//! coverage while giving up the payload-regime claim it exists for.
//!
//! # Why warm-up turns
//!
//! `ColdTierWalk::capture_chain` stops at the first block the bounded writer
//! queue refuses, so one turn persists only the first handful of blocks — far
//! short of the 64 blocks a 1024-token boundary needs. Blocks already on disk
//! are skipped without re-enqueueing, so the frontier advances every turn; the
//! warm-up dial simply runs enough turns for it to pass the target boundary.
//! Measured on qwen3 with a 4.8k-token prompt: restorable prefix grew
//! 176 -> 560 -> 944 tokens across three runs.
//!
//! Gated on `MLX_TEST_MODEL_PATH`. The tier manager is a process-global
//! `OnceLock`, so this must be the only thing in the process touching it —
//! hence `#[ignore]` plus `--test-threads=1`. Both gates in this binary are
//! safe to run in one process: `run_restart_parity` honours an already-created
//! root, the cold keys are content-derived (different prompts => disjoint
//! chains), and every stats assertion is a delta around its own instance 2.
//!
//! That claim also depends on the tier ROOT outliving gate 1, which is why the
//! harness no longer deletes it in teardown. It used to. Without an inherited
//! `MLX_COLD_CACHE_DIR` the root is `temp_dir()/mlx-cold-parity-{pid}`, so gate
//! 2 recreated the same pathname while the live manager still held a descriptor
//! for the directory gate 1 had unlinked — and a descriptor to an unlinked
//! directory fails every name lookup with ENOENT. Gate 2 stored nothing,
//! restored nothing, and blamed the restore path. Every invocation below passes
//! `MLX_COLD_CACHE_DIR` explicitly, which takes the inherited arm and never hit
//! it; running the two gates with no cache dir set is what did.
//! They are slow enough (~4 min per fresh load, dominated by full-shard weight
//! hashing) that running them separately is usually what you want.
//!
//! The whole-window gate needs nothing but the model path:
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//!     MLX_TEST_MODEL_PATH=~/.mlx-node/models/Gemma-4-26B-A4B-IT-UD-Q3_K_XL-mlx \
//!     cargo test -p mlx-core --release --test gemma4_cold_tier_parity \
//!     -- --ignored --exact --nocapture gemma4_cold_tier_restart_parity
//! ```
//!
//! The sub-window gate additionally reads the inference trace, which must be
//! turned on by the invocation — see `gate_inference_trace_path`:
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//!     MLX_INFERENCE_TRACE=1 \
//!     MLX_INFERENCE_TRACE_FILE=$(mktemp -t gemma4-sub-window).trace \
//!     MLX_TEST_MODEL_PATH=~/.mlx-node/models/Gemma-4-26B-A4B-IT-UD-Q3_K_XL-mlx \
//!     cargo test -p mlx-core --release --test gemma4_cold_tier_parity \
//!     -- --ignored --exact --nocapture gemma4_cold_tier_restart_parity_sub_window
//! ```

mod cold_tier_parity_harness;

use std::path::PathBuf;

use cold_tier_parity_harness as harness;
use mlx_core::models::gemma4::Gemma4Model;

/// Gemma4's `sliding_window` on every shipped 12B/26B/31B checkpoint. The
/// sidecar payload carries `min(boundary, window)` rows, so this is the line
/// between the two gates below — not a floor on what is representable.
const SLIDING_WINDOW_TOKENS: u32 = 1024;

/// Path the inference trace is being written to, per the environment this test
/// binary was *invoked* with.
///
/// Read-only on purpose. `mlx_core::inference_trace` latches
/// `MLX_INFERENCE_TRACE` / `MLX_INFERENCE_TRACE_FILE` into a process-global
/// `OnceLock` the first time anything traces, so the channel has to exist
/// before the first model load — but setting it from inside a
/// `#[tokio::test(flavor = "multi_thread")]` body would mean `set_var` on a
/// process that already has live threads, which is exactly the unsound case.
/// Supplying it on the command line gets the same guarantee with none of the
/// hazard, and keeps this helper from reaching into a sibling gate's
/// environment.
///
/// The sub-window gate needs the channel because it is the only place outside
/// the crate where `state=cold_sidecar` and the zero-replay signal are
/// observable, so a missing trace file is a setup error, not a skip: an
/// unobserved gate that "passes" is worse than one that refuses to run.
fn gate_inference_trace_path() -> PathBuf {
    let Some(path) = std::env::var_os("MLX_INFERENCE_TRACE_FILE") else {
        panic!(
            "this gate reads the inference trace and cannot assert anything without it.\n\
             Re-run with the channel supplied by the invocation, e.g.\n\
             \x20 MLX_INFERENCE_TRACE=1 \\\n\
             \x20 MLX_INFERENCE_TRACE_FILE=/tmp/gemma4-sub-window.trace \\\n\
             \x20 MLX_TEST_MODEL_PATH=<checkpoint> \\\n\
             \x20 cargo test -p mlx-core --release --test gemma4_cold_tier_parity \\\n\
             \x20   -- --ignored --exact --test-threads=1 \\\n\
             \x20   gemma4_cold_tier_restart_parity_sub_window"
        );
    };
    // Mirrors `inference_trace::env_flag_value_enabled`, which is `pub(crate)`
    // and so out of reach from an integration test. A pointing file with the
    // flag off yields an empty channel and assertions that fail for the wrong
    // reason 30 minutes later, so reject it here.
    let flag = std::env::var("MLX_INFERENCE_TRACE").unwrap_or_default();
    assert!(
        matches!(
            flag.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        "MLX_INFERENCE_TRACE_FILE is set but MLX_INFERENCE_TRACE={flag:?} leaves the channel off; \
         set MLX_INFERENCE_TRACE=1"
    );
    PathBuf::from(path)
}

/// Every trace line containing `needle`.
fn trace_lines<'a>(trace: &'a str, needle: &str) -> Vec<&'a str> {
    trace.lines().filter(|line| line.contains(needle)).collect()
}

/// Value of a `key=<u32>` field in one trace line.
fn trace_field(line: &str, key: &str) -> Option<u32> {
    line.split_whitespace()
        .find_map(|token| token.strip_prefix(key)?.parse().ok())
}

/// A prompt that clears one sliding window but stays inside the second, so the
/// only interval-cadence checkpoint prefill records sits at exactly
/// `sliding_window` — see the module doc's floor/ceiling discussion.
///
/// Built rather than written out: the paragraphs are numbered so the text is not
/// a degenerate repetition (which would make the prefix-cache identity and the
/// sliding state uninteresting), and leaked because `ColdTierParitySpec::prompt`
/// is a `&'static str`. Measured on this checkpoint's tokenizer, each paragraph
/// is ~75 tokens, so 16 of them land near 1.2k — past the 1024-token window,
/// still inside the second one.
fn one_window_prompt() -> &'static str {
    notes_prompt(16)
}

/// A prompt comfortably INSIDE one sliding window — the shape of an ordinary
/// chat turn, and the case the old `boundary >= window` rule could never back.
///
/// Four paragraphs at ~75 tokens each land near 350 tokens once the chat
/// template wraps them: roughly 21 blocks of 16, a third of a window.
fn sub_window_prompt() -> &'static str {
    notes_prompt(4)
}

/// `count` numbered notes wrapped in a short instruction, leaked to `'static`.
fn notes_prompt(count: u32) -> &'static str {
    let mut prompt =
        String::from("Answer with a single short paragraph. First, read these notes.\n\n");
    for index in 1..=count {
        prompt.push_str(&format!(
            "Note {index}: when a block-paged key-value cache persists warm prefixes to local \
             solid-state storage, the engineer reviewing revision {index} must weigh the block \
             size, the eviction policy, the checksum cost, and the exact boundary at which any \
             out-of-pool recurrent or sliding-window state can be resumed soundly rather than \
             guessed.\n",
        ));
    }
    prompt.push_str("\nNow summarize the single most important tradeoff in one sentence.");
    Box::leak(prompt.into_boxed_str())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Gemma4 checkpoint; run with --test-threads=1"]
async fn gemma4_cold_tier_restart_parity() {
    harness::run_restart_parity(
        harness::ColdTierParitySpec::new("gemma4")
            .with_prompt(one_window_prompt())
            // The pool covers only the full-attention layers; this is ample for
            // a ~1.3k-token prompt plus the decode tail.
            .with_pool_memory_mb(1024)
            // Advance the persisted K/V chain past one whole window (64 blocks
            // at block_size 16) before the measured restore. See the module doc.
            .with_capture_warmup_turns(12)
            // See the module doc: with a `ColdSidecarPolicy` installed, a
            // restore this deep in a fresh instance is only reachable through a
            // validated sliding-window sidecar, and this fixture's shallowest
            // checkpoint sits at exactly one window.
            .with_min_restored_tokens(SLIDING_WINDOW_TOKENS)
            // `install_gemma4_sliding_cold_sidecar` has six `Ok(None)` arms and
            // each falls through to a full `run_sliding_only_prefill` replay
            // whose output is identical, so nothing else here can see the
            // difference.
            .expecting_sidecar_install(),
        |model_dir, messages, config| async move {
            // Loaded fresh per instance and dropped when this future
            // completes, so instance 2 really starts from an empty hot cache.
            let model = Gemma4Model::load_from_dir(&model_dir.to_string_lossy(), None).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}

/// The sub-window gate: a prompt SHORTER than one sliding window must produce a
/// genuine sidecar-backed restore, with no replay and no latch.
///
/// # What the extra assertions buy
///
/// The shared harness can only see `ChatResult`. On a sub-window prompt that is
/// not enough, because two very different implementations agree on it:
///
/// * the design this change implements — the sidecar backs the whole reused
///   prefix, `install_gemma4_sliding_cold_sidecar` seats the rotating caches at
///   the boundary, and not one token is re-run through the decoder;
/// * the fall-through design that was REJECTED — restore the K/V prefix anyway,
///   leave the sliding half unbacked, and pay `run_sliding_only_prefill` over
///   the whole prefix to reconstruct it.
///
/// Both produce identical text, identical `num_tokens` and identical
/// `cached_tokens`. What separates them is `PagedKVCacheAdapter`'s
/// `aux_prefix_unbacked` latch, which is set when the restored sidecar does not
/// cover exactly `cached_token_count` and then discharged by a replay. That
/// field is `pub(crate)` and, either way, is `false` again by the time the turn
/// ends — so the only faithful observation point is the inference trace, which
/// records the decision as it is made:
///
/// ```text
/// gemma4 sliding_prefix_prepare_done state=cold_sidecar cached_prefix_tokens=B primed_prefix_tokens=B replay_delta_tokens=0
/// gemma4 paged_prefill_sliding_prefix_skipped cached_prefix_tokens=B sliding_primed_prefix_tokens=B reason=already_primed
/// ```
///
/// `primed_prefix_tokens` IS the restored sidecar's `layout.boundary_tokens`
/// (`install_gemma4_sliding_cold_sidecar` returns it verbatim) and
/// `cached_prefix_tokens` IS the adapter's `cached_token_count`, so asserting
/// they are equal is assertion (ii) exactly — and `aux_prefix_state_missing`
/// compares those same two numbers, so it is assertion (iii) exactly too. The
/// `_skipped ... reason=already_primed` line is the zero-replay half: its
/// alternative, `paged_prefill_sliding_prefix_done ... replay_tokens=N`, is
/// emitted only when `run_sliding_only_prefill` actually ran.
///
/// # Fixture caveat: prompt length vs block alignment
///
/// Below one window the ONLY checkpoint a capture can anchor on is the
/// prompt-boundary one, which sits at `floor(prompt_len / block_size) *
/// block_size`. Restore is capped at `prompt_len - 1` (vLLM's
/// `max_cache_hit_tokens` rule), so when `prompt_len` happens to be an exact
/// multiple of `block_size` the only sub-window sidecar sits one block ABOVE
/// anything the restore can ask for, and nothing is backed. That is a 1-in-16
/// property of how this prompt tokenizes, not a code path — if this gate fails
/// with `cached_tokens = 0`, check `request_tokens=` / `prompt_boundary=` on
/// the `sliding_cold_sidecar_capture_skipped` trace line and nudge the prompt
/// by a word.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Gemma4 checkpoint; run with --test-threads=1"]
async fn gemma4_cold_tier_restart_parity_sub_window() {
    let trace_path = gate_inference_trace_path();

    harness::run_restart_parity(
        harness::ColdTierParitySpec::new("gemma4-sub-window")
            .with_prompt(sub_window_prompt())
            .with_pool_memory_mb(1024)
            // ~21 blocks to cover. The chain advanced ~24 blocks/turn when this
            // was measured on qwen3, so this is several times over.
            .with_capture_warmup_turns(6)
            // The inspector below reads a trace line that names the state, so
            // this gate was already the one family gate that could see an
            // install. Assert it structurally too, so it does not depend on
            // whether the stream path happens to emit that line.
            .expecting_sidecar_install()
            .with_restore_inspector(move |observation| {
                assert_sub_window_sidecar_restore(observation, &trace_path);
            }),
        |model_dir, messages, config| async move {
            let model = Gemma4Model::load_from_dir(&model_dir.to_string_lossy(), None).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}

/// Assertions (i)-(iii) of the sub-window gate; (iv) is the harness's own.
fn assert_sub_window_sidecar_restore(
    observation: &harness::RestoreObservation<'_>,
    trace_path: &std::path::Path,
) {
    let family = observation.family;
    let trace = observation.trace;
    assert!(
        !trace.is_empty(),
        "[{family}] the inference-trace observation channel is dead: nothing was appended to \
         {} while the restore instance ran, so none of the sidecar assertions below can mean \
         anything. Both trace vars were accepted at entry, which leaves two causes: the path is \
         not writable by this process, or something traced before this gate and latched a \
         different destination into `inference_trace`'s process-global OnceLock.",
        trace_path.display()
    );

    // One preparation per turn. A SECOND one means
    // `suppress_large_sliding_prefix_reuse_if_needed` fired and re-prepared
    // from zero, which only happens when the first preparation left a replay
    // delta — i.e. the sidecar did not back the prefix.
    let prepared = trace_lines(trace, "sliding_prefix_prepare_done");
    assert_eq!(
        prepared.len(),
        1,
        "[{family}] the restore turn prepared its sliding prefix {} times; more than once means \
         the first preparation left a replay delta and the large-reuse suppressor restarted the \
         turn cold:\n{}",
        prepared.len(),
        prepared.join("\n")
    );
    let prepared = prepared[0];

    // ---- (i) a sub-window prefix really came back, from the sidecar --------
    assert!(
        prepared.contains("state=cold_sidecar"),
        "[{family}] the restore instance did not prime its sliding state from the cold sidecar. \
         A fresh instance has no in-memory checkpoint and no live cache, so any other state here \
         means the sidecar was not selected and the sliding half was rebuilt by replay — exactly \
         the fall-through this gate exists to rule out.\n  saw: {prepared}\n  trace: {}\n  If \
         cached_tokens is 0, grep that file for `sliding_cold_sidecar_capture_skipped` — see this \
         test's fixture caveat about prompt length vs block alignment.",
        trace_path.display()
    );

    let boundary = trace_field(prepared, "cached_prefix_tokens=").unwrap_or_else(|| {
        panic!("[{family}] no cached_prefix_tokens= field in: {prepared}");
    });
    let primed = trace_field(prepared, "primed_prefix_tokens=").unwrap_or_else(|| {
        panic!("[{family}] no primed_prefix_tokens= field in: {prepared}");
    });
    let replay_delta = trace_field(prepared, "replay_delta_tokens=").unwrap_or_else(|| {
        panic!("[{family}] no replay_delta_tokens= field in: {prepared}");
    });

    assert!(
        boundary > 0,
        "[{family}] the restore instance resumed from nothing. Under the old \
         `boundary >= window` rule this was the ONLY possible answer for a prompt shorter than \
         {SLIDING_WINDOW_TOKENS} tokens, and it is the delta this change is about.\n  {prepared}"
    );
    assert_eq!(
        boundary, observation.result.cached_tokens,
        "[{family}] the traced cached prefix ({boundary}) and the reported one ({}) disagree",
        observation.result.cached_tokens
    );
    assert!(
        boundary < SLIDING_WINDOW_TOKENS,
        "[{family}] this gate is only meaningful below one sliding window, but the restore came \
         back at {boundary} >= {SLIDING_WINDOW_TOKENS}. The prompt grew past a window — shrink it, \
         or the case is already covered by the long-prompt gate.\n  {prepared}"
    );
    assert_eq!(
        boundary % harness::DEFAULT_BLOCK_SIZE,
        0,
        "[{family}] a sidecar boundary must be block-aligned, got {boundary}"
    );

    // ---- (ii)/(iii) the sidecar backs EXACTLY the reused prefix ------------
    // `primed_prefix_tokens` is the restored sidecar's `layout.boundary_tokens`
    // and `cached_prefix_tokens` is `cached_token_count`; those are the two
    // numbers `aux_prefix_state_missing` compares, so equality here is both
    // "layout.boundary_tokens == cached_token_count" and
    // "aux_prefix_unbacked() == false".
    assert_eq!(
        primed, boundary,
        "[{family}] the restored sidecar backs {primed} tokens but the request resumes from \
         {boundary}. That is exactly the condition `aux_prefix_state_missing` latches, so the \
         adapter would have flagged the prefix unbacked and the delta would have been replayed \
         through the decoder — a silent replay has crept in.\n  {prepared}"
    );
    assert_eq!(
        replay_delta, 0,
        "[{family}] the restore left {replay_delta} tokens of sliding state to rebuild\n  \
         {prepared}"
    );

    // ---- zero replay, corroborated on the prefill side ---------------------
    let replayed = trace_lines(trace, "paged_prefill_sliding_prefix_done");
    assert!(
        replayed.is_empty(),
        "[{family}] `run_sliding_only_prefill` ran during the restore turn, so the sliding half \
         was rebuilt from token ids rather than resumed from the sidecar:\n{}",
        replayed.join("\n")
    );
    let skipped = trace_lines(trace, "paged_prefill_sliding_prefix_skipped");
    assert_eq!(
        skipped.len(),
        1,
        "[{family}] expected exactly one already-primed prefill skip, saw {}:\n{}",
        skipped.len(),
        skipped.join("\n")
    );
    assert!(
        skipped[0].contains("reason=already_primed")
            && trace_field(skipped[0], "sliding_primed_prefix_tokens=") == Some(boundary),
        "[{family}] the prefill did not skip the sliding replay for the whole {boundary}-token \
         prefix:\n  {}",
        skipped[0]
    );

    eprintln!(
        "[{family}] sub-window cold sidecar PASS: boundary={boundary} (< window \
         {SLIDING_WINDOW_TOKENS}), replay_delta=0, prefill skipped as already_primed"
    );
}

/// Offline cover for [`assert_sub_window_sidecar_restore`].
///
/// The gate it serves needs 14 GB of real weights and the better part of an
/// hour, so the assertions themselves would otherwise be unverified code — and
/// an assertion that cannot fail is worse than none. These feed it hand-built
/// traces whose lines are copied from the `write_inference_trace` format
/// strings in `models::gemma4::model`, so a reworded trace line breaks a
/// one-second test instead of silently turning the gate into a rubber stamp.
///
/// Each mutant below is a real alternative implementation, not a typo: a
/// sidecar that backs less than the reused prefix (the latch condition), a
/// restore that replayed the sliding half anyway (the rejected fall-through),
/// and a boundary that never went below the window (the gate testing nothing).
#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::engine::types::ChatResult;

    const BOUNDARY: u32 = 336;

    fn chat_result(cached_tokens: u32) -> ChatResult {
        ChatResult {
            text: "ok".to_string(),
            tool_calls: Vec::new(),
            thinking: None,
            num_tokens: 8,
            prompt_tokens: 350,
            reasoning_tokens: 0,
            finish_reason: "stop".to_string(),
            raw_text: "ok".to_string(),
            cached_tokens,
            performance: None,
        }
    }

    /// A restore turn that went the way this change intends, with enough
    /// neighbouring lines that the filters have something to reject.
    fn backed_trace(cached: u32, primed: u32) -> String {
        format!(
            "[MLX_TRACE] gemma4 paged_adapter_prepare live=false request_tokens=0 \
             common_prefix=0 total_budget=350 reuse_cache=true\n\
             [MLX_TRACE] gemma4 paged_adapter_prepare_done reason=ColdRestore \
             cached_prefix_tokens={cached} cached_blocks=21 allocated_blocks=22 \
             request_tokens=350 blocks=22 continued_live=false\n\
             [MLX_TRACE] gemma4 sliding_prefix_prepare_done state=cold_sidecar \
             cached_prefix_tokens={cached} primed_prefix_tokens={primed} \
             replay_delta_tokens={delta} elapsed_ms=4.2\n\
             [MLX_TRACE] gemma4 paged_prefill_start full_tokens=350 \
             cached_prefix_tokens={cached} suffix_tokens=14\n\
             [MLX_TRACE] gemma4 paged_prefill_sliding_prefix_skipped \
             cached_prefix_tokens={cached} sliding_primed_prefix_tokens={primed} \
             reason=already_primed\n",
            delta = cached.saturating_sub(primed)
        )
    }

    fn inspect(trace: &str, cached_tokens: u32) {
        let result = chat_result(cached_tokens);
        assert_sub_window_sidecar_restore(
            &harness::RestoreObservation {
                family: "gemma4-sub-window",
                result: &result,
                trace,
            },
            std::path::Path::new("/nonexistent/trace"),
        );
    }

    #[test]
    fn accepts_a_sub_window_sidecar_that_backs_the_whole_prefix() {
        inspect(&backed_trace(BOUNDARY, BOUNDARY), BOUNDARY);
    }

    #[test]
    #[should_panic(expected = "observation channel is dead")]
    fn rejects_a_dead_trace_channel() {
        inspect("", BOUNDARY);
    }

    #[test]
    #[should_panic(expected = "did not prime its sliding state from the cold sidecar")]
    fn rejects_a_restore_primed_from_anything_but_the_sidecar() {
        let trace = backed_trace(BOUNDARY, BOUNDARY).replace("state=cold_sidecar", "state=replay");
        inspect(&trace, BOUNDARY);
    }

    #[test]
    #[should_panic(expected = "resumed from nothing")]
    fn rejects_the_pre_fix_behaviour_of_restoring_no_prefix_at_all() {
        inspect(&backed_trace(0, 0), 0);
    }

    /// The latch condition: sidecar shallower than the reused prefix, which is
    /// what `aux_prefix_state_missing` compares.
    #[test]
    #[should_panic(expected = "a silent replay has crept in")]
    fn rejects_a_sidecar_that_backs_less_than_the_reused_prefix() {
        inspect(&backed_trace(BOUNDARY, BOUNDARY - 64), BOUNDARY);
    }

    /// The rejected fall-through: prefix restored, sliding half rebuilt by
    /// `run_sliding_only_prefill`. Text and `cached_tokens` are unchanged, so
    /// only the prefill line separates it from a pass.
    #[test]
    #[should_panic(expected = "was rebuilt from token ids")]
    fn rejects_a_restore_that_replayed_the_sliding_half() {
        let trace = backed_trace(BOUNDARY, BOUNDARY).replace(
            "paged_prefill_sliding_prefix_skipped cached_prefix_tokens=336 \
             sliding_primed_prefix_tokens=336 reason=already_primed",
            "paged_prefill_sliding_prefix_done cached_prefix_tokens=336 \
             restored_prefix_tokens=0 replay_tokens=336 checkpoint_stored=true",
        );
        inspect(&trace, BOUNDARY);
    }

    /// A prompt that grew past the window would make this gate a duplicate of
    /// its long-prompt sibling rather than a failure, so it is refused.
    #[test]
    #[should_panic(expected = "only meaningful below one sliding window")]
    fn rejects_a_boundary_at_or_above_the_window() {
        inspect(
            &backed_trace(SLIDING_WINDOW_TOKENS, SLIDING_WINDOW_TOKENS),
            SLIDING_WINDOW_TOKENS,
        );
    }

    #[test]
    #[should_panic(expected = "the traced cached prefix")]
    fn rejects_a_reported_prefix_that_disagrees_with_the_trace() {
        inspect(&backed_trace(BOUNDARY, BOUNDARY), BOUNDARY - 16);
    }

    #[test]
    fn trace_field_reads_only_its_own_key() {
        let line = "[MLX_TRACE] gemma4 paged_prefill_sliding_prefix_skipped \
                    cached_prefix_tokens=336 sliding_primed_prefix_tokens=336 \
                    reason=already_primed";
        assert_eq!(trace_field(line, "cached_prefix_tokens="), Some(336));
        assert_eq!(
            trace_field(line, "sliding_primed_prefix_tokens="),
            Some(336)
        );
        assert_eq!(trace_field(line, "primed_prefix_tokens="), None);
        assert_eq!(trace_field(line, "absent="), None);
    }
}
