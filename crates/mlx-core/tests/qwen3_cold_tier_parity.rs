//! Real-weights cold-tier restart-parity gate for Qwen3 (dense).
//!
//! Validates the full A1–A4 cold-tier pipeline end to end at the model level:
//! a persist-enabled instance captures its full paged KV blocks to the SSD
//! cold tier on turn finalize (A3), and a SECOND fresh instance — standing in
//! for a process restart, with an empty in-memory hot cache — restores that
//! persisted prefix through `find_cached_prefix` (A4) and reports it in
//! `cached_tokens`. The restored greedy decode must be byte-identical to the
//! original, and to a THIRD instance that runs the same prompt with
//! persistence disabled.
//!
//! Qwen3 dense is the only family currently in
//! `cold_tier::COLD_RESTORE_FAMILIES`, because its pool covers ALL layers, so
//! a restored block reconstructs the whole prefix. The scenario itself lives
//! in [`cold_tier_parity_harness`] so every family joining the allowlist is
//! gated by the same three-instance comparison, the same restore-engaged
//! floor, and the same zero-corruption check.
//!
//! Gated on `MLX_TEST_MODEL_PATH`: a plain `cargo test --ignored` without it
//! still passes (the harness short-circuits before any load). The tier root
//! comes from `MLX_COLD_CACHE_DIR` when set, else a per-process temp dir.
//! Because the tier manager is a process-global `OnceLock`, this must be the
//! only thing in the process touching it — hence `#[ignore]` plus
//! `--test-threads=1`.
//!
//! Run locally with:
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//!     MLX_TEST_MODEL_PATH=~/.mlx-node/models/qwen3-0.6b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_cold_tier_parity \
//!     -- --ignored --test-threads=1 --nocapture
//! ```

mod cold_tier_parity_harness;

use cold_tier_parity_harness as harness;
use mlx_core::models::qwen3::persistence::load_with_thread as qwen3_load_with_thread;

// ---------------------------------------------------------------------------
// Cold-tier restart parity gate
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3 checkpoint; run with --test-threads=1"]
async fn qwen3_cold_tier_restart_parity() {
    harness::run_restart_parity(
        harness::ColdTierParitySpec::new("qwen3"),
        |model_dir, messages, config| async move {
            // Loaded fresh per instance and dropped when this future
            // completes, so instance 2 really starts from an empty hot cache.
            let model = qwen3_load_with_thread(&model_dir.to_string_lossy()).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}
