//! Real-weights cold-tier restart-parity gate for Qwen3.5 dense (hybrid GDN).
//!
//! Same three-instance scenario as the qwen3 gate — see
//! [`cold_tier_parity_harness`] — but qwen3_5 is a HYBRID family, and that
//! changes what a pass proves.
//!
//! # Why a pass here is GDN-sidecar evidence, not just KV evidence
//!
//! Qwen3.5 sizes its paged pool over the FULL-ATTENTION layers only; the
//! interleaved GDN (gated delta-net) layers keep their recurrent state in a
//! per-layer `ArraysCache` (conv + recurrent) OUTSIDE the pool. So its
//! `ColdTierContext` carries a `ColdSidecarPolicy`, and with a policy installed
//! `ColdTierWalk::restore_extend` restores NOTHING unless a validated GDN
//! sidecar backs the boundary (`deepest_backed_boundary` -> `None` =>
//! `ColdRestore::miss()`).
//!
//! Instance 2 is a freshly loaded model, so its in-process prefix cache is
//! empty and a hot hit is impossible. Any non-zero `cached_tokens` it reports
//! can therefore only have come through the reconcile-down walk — which means a
//! GDN state sidecar was found on disk, decoded, and validated against this
//! checkpoint's geometry. That is what turns "the tier restored something" into
//! "the tier restored the recurrent half too".
//!
//! # Why warm-up turns
//!
//! Unlike gemma4's sliding sidecar (representable only at a whole 1024-token
//! window), the GDN sidecar sits at a SINGLE `gdn_checkpoint_target` boundary —
//! the largest full block strictly before the end of the prompt, a handful of
//! blocks for `DEFAULT_PROMPT`. But `ColdTierWalk::capture_chain` still stops at
//! the first block the bounded writer queue refuses, so one turn persists only
//! the first handful of K/V blocks. The GDN sidecar is only WRITTEN once the
//! persisted K/V chain reaches its boundary (`cold_captured_blocks`), so a few
//! warm-up turns deepen the chain past it. Blocks already on disk are skipped
//! without re-enqueueing, so the frontier advances every turn.
//!
//! Gated on `MLX_TEST_MODEL_PATH`. The tier manager is a process-global
//! `OnceLock`, so this must be the only thing in the process touching it —
//! hence `#[ignore]` plus `--test-threads=1`.
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//!     MLX_TEST_MODEL_PATH=~/.mlx-node/models/qwen3.5-0.8b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_5_cold_tier_parity \
//!     -- --ignored --test-threads=1 --nocapture
//! ```

mod cold_tier_parity_harness;

use cold_tier_parity_harness as harness;
use mlx_core::models::qwen3_5::persistence::load_with_thread as qwen3_5_load_with_thread;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 dense checkpoint; run with --test-threads=1"]
async fn qwen3_5_cold_tier_restart_parity() {
    harness::run_restart_parity(
        harness::ColdTierParitySpec::new("qwen3_5")
            // The pool covers only the full-attention layers, so this is ample
            // for `DEFAULT_PROMPT` plus a short decode tail.
            .with_pool_memory_mb(512)
            // Deepen the persisted K/V chain past the GDN sidecar's
            // `gdn_checkpoint_target` boundary (a few blocks for this prompt)
            // before the measured restore. See the module doc.
            .with_capture_warmup_turns(3),
        |model_dir, messages, config| async move {
            // Loaded fresh per instance and dropped when this future
            // completes, so instance 2 really starts from an empty hot cache.
            let model = qwen3_5_load_with_thread(&model_dir.to_string_lossy()).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}
