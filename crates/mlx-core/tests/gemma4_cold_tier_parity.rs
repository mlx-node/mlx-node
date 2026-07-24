//! Real-weights cold-tier restart-parity gate for Gemma4 (hybrid).
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
//! this checkpoint's geometry. That is why the restore floor below is raised to
//! a whole `sliding_window`: the sidecar layout is only representable at
//! `boundary >= window` (see `models::gemma4::sliding_sidecar`), so a
//! sidecar-backed restore is necessarily at least that deep, and asserting it
//! turns "the tier restored something" into "the tier restored the auxiliary
//! half too".
//!
//! # Why the prompt is sized the way it is
//!
//! Two opposing constraints pin it into a narrow band.
//!
//! **Floor.** The shared `DEFAULT_PROMPT` is a few hundred tokens — comfortably
//! BELOW gemma4's 1024-token sliding window. Under the `boundary >= window` rule
//! that prompt would restore zero sidecar-backed prefix and still pass on the
//! K/V path alone, i.e. it would flip a family's allowlist entry on evidence
//! that never touched the sliding code. So the prompt must clear one window.
//!
//! **Ceiling.** `gemma4_sliding_prefix_checkpoint_limit` bounds the in-memory
//! checkpoint store by a memory budget — on a 26B checkpoint that works out to
//! TWO entries — and prefill records one checkpoint per
//! `gemma4_sliding_decode_checkpoint_interval` (= one window) crossed. A prompt
//! several windows long therefore retains only its DEEPEST two checkpoints
//! (e.g. 3072 and 4096) and evicts the one at 1024. Since the capture can only
//! anchor at a boundary the persisted K/V chain also reaches — and that chain
//! advances by one writer-queue's worth of blocks per turn — the deep
//! checkpoints are unreachable and the shallow one is gone. Keeping the prompt
//! inside the SECOND window means the only interval checkpoint is the one at
//! `sliding_window`, which is both retained and the first boundary a growing
//! chain can cover.
//!
//! # Why warm-up turns
//!
//! `ColdTierWalk::capture_chain` stops at the first block the bounded writer
//! queue refuses, so one turn persists only the first handful of blocks — far
//! short of the 64 blocks a 1024-token boundary needs. Blocks already on disk
//! are skipped without re-enqueueing, so the frontier advances every turn; the
//! warm-up dial simply runs enough turns for it to pass one window. Measured on
//! qwen3 with a 4.8k-token prompt: restorable prefix grew 176 -> 560 -> 944
//! tokens across three runs.
//!
//! Gated on `MLX_TEST_MODEL_PATH`. The tier manager is a process-global
//! `OnceLock`, so this must be the only thing in the process touching it —
//! hence `#[ignore]` plus `--test-threads=1`.
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//!     MLX_TEST_MODEL_PATH=~/.mlx-node/models/Gemma-4-26B-A4B-IT-UD-Q3_K_XL-mlx \
//!     cargo test -p mlx-core --test gemma4_cold_tier_parity \
//!     -- --ignored --test-threads=1 --nocapture
//! ```

mod cold_tier_parity_harness;

use cold_tier_parity_harness as harness;
use mlx_core::models::gemma4::Gemma4Model;

/// Gemma4's `sliding_window` on every shipped 12B/26B/31B checkpoint. The
/// sidecar's fixed-length payload is only representable at boundaries at least
/// this deep, so it doubles as the restore floor this gate asserts.
const SLIDING_WINDOW_TOKENS: u32 = 1024;

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
    let mut prompt =
        String::from("Answer with a single short paragraph. First, read these notes.\n\n");
    for index in 1..=16u32 {
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
            // validated sliding-window sidecar.
            .with_min_restored_tokens(SLIDING_WINDOW_TOKENS),
        |model_dir, messages, config| async move {
            // Loaded fresh per instance and dropped when this future
            // completes, so instance 2 really starts from an empty hot cache.
            let model = Gemma4Model::load_from_dir(&model_dir.to_string_lossy(), None).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}
