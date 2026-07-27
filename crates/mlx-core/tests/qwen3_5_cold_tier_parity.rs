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
//! checkpoint's geometry.
//!
//! Be precise about the last step, because for a long time this file was not.
//! Found + read + checksummed + layout-validated is NOT the same as INSTALLED.
//! `install_dense_gdn_cold_sidecar` has eight `Ok(false)` arms and every one of
//! them falls through to `replay_materialized` — a full O(prefix) GDN replay
//! that reconstructs CORRECT recurrent state. Under that fall-through
//! `cached_tokens`, `hits`, `corruptions` and the text/`num_tokens` parity are
//! all exactly what a healthy run reports, so nothing else in the harness moves.
//! `expecting_sidecar_install()` is what closes that: it asserts the
//! `installed` sidecar counter advanced during instance 2 alone. Only with it
//! set does a pass here mean "the tier restored the recurrent half too" rather
//! than "the tier could have".
//!
//! # Why warm-up turns
//!
//! The GDN sidecar may sit at any rung of the prefill's checkpoint
//! ladder (`gdn_prefill_checkpoint_boundaries`): the deepest rung is
//! `gdn_checkpoint_target` — the largest full block strictly before the end of
//! the prompt — and each shallower rung is a block-aligned quarter of the one
//! above it. But `ColdTierWalk::capture_chain` still stops at the first block
//! the bounded writer queue refuses, so one turn persists only the first
//! handful of K/V blocks. A sidecar is only WRITTEN at the deepest rung the
//! persisted K/V chain already covers (`cold_captured_blocks`), so a few
//! warm-up turns deepen the chain until a rung qualifies. Blocks already on
//! disk are skipped without re-enqueueing, so the frontier advances every turn.
//!
//! # Why the capture and restore prompts differ
//!
//! Everything above is about a LADDER, but a gate whose three instances all run
//! a ~90-token prompt cannot see one. At 90 tokens the ladder is `[16, 80]` and
//! one warm-up turn's chain reach (~8 blocks = 128 tokens) already covers the
//! deepest rung, so the restore always anchors at the prompt's end — the
//! shallow rungs are never exercised and collapsing the ladder to a single
//! endpoint boundary leaves such a gate green.
//!
//! So this fixture uses `harness::ladder_capture_prompt` / `ladder_restore_prompt`:
//! a long shared body and then a diverging tail. Two things follow.
//! The capture prompt's deepest rung sits far past what a bounded writer queue
//! drains in a few turns. And the divergence caps `kv_chain_upper_bound` —
//! which walks the block chain and breaks at the first key not on disk — well
//! below that rung, whatever the machine's I/O did. The restore therefore MUST
//! land on a shallower rung, which is what `harness`'s assertion 1b checks.
//!
//! Observed on `qwen3.5-0.8b-mlx-bf16`:
//!
//! ```text
//! capture prompt 1259 tok -> ladder [16, 64, 304, 1248]   (deepest = 78 blocks)
//! warm-up 1 cached=0     warm-up 2 cached=64     instance 1 cached=64
//! instance 2 (restart)   cached=304   <-- rung 3 of 4, strictly below 1248
//! sidecar telemetry      boundary_skips=0 already_persisted=1 enqueued=1
//! ```
//!
//! `already_persisted=1` alongside `enqueued=1` is the healthy steady state,
//! not a miss: one turn wrote rung 304 and the next re-selected the same chain
//! and deduped. It has its own counter precisely so that `enqueued=0` in a
//! later window cannot be misread as a collapsed ladder.
//!
//! Honest limit: this kills a ladder collapsed to ONE rung. It does not kill a
//! collapse to two (`[304, 1248]` still restores at 304), and it does not kill
//! a RATIO change either: 4 = 2^2, so halving the spacing produces a ladder
//! that still contains the rung a ~24-block chain would have anchored on
//! anyway. The claim being gated is "a rung strictly below the prompt end
//! exists and is used", not "there are exactly four, spaced by four".
//!
//! Neither gap is left open. Both are pinned by exact rung VALUES in
//! `qwen3_5::paged_forward::gdn_checkpoint_tests::ladder_rungs_are_quarters_of_the_one_above`
//! — model-free, milliseconds, and it fails on every ratio and rung-count
//! mutation. That is the right division of labour: a multi-minute
//! three-model-load gate should prove the thing only real weights can prove
//! (that a shallow rung is genuinely WRITTEN, found, decoded and restored
//! across a process boundary), and arithmetic should be pinned by arithmetic.
//!
//! One consequence worth stating plainly: when the ladder collapses to one
//! rung, the failure here is NOT "the restore picked the wrong rung". Nothing
//! is written at all, so instance 2 restores zero and the harness's assertion 1
//! fires — a message about the restore path, for a defect in the prefill. The
//! harness prints the computed ladder and the sidecar telemetry BEFORE that
//! assertion for exactly this reason.
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
            // for two ~1260-token prompts plus a short decode tail. Raised from
            // 512 with the prompt: a pool-pressure eviction of the shallow
            // sidecar this gate depends on would read as a restore failure.
            .with_pool_memory_mb(1024)
            .with_prompt(harness::ladder_capture_prompt())
            .with_restore_prompt(harness::ladder_restore_prompt())
            // Two turns put the persisted chain past a mid ladder rung. Fewer
            // than the three the short prompt needed, because the chain no
            // longer has to reach the prompt's own end — only a shallow rung.
            .with_capture_warmup_turns(2)
            // Without this the module doc above overclaims: every other
            // assertion here is satisfied by a restart that reads and validates
            // the GDN sidecar and then discards it, falling through to a full
            // O(prefix) recurrent replay whose output is identical.
            .expecting_sidecar_install(),
        |model_dir, messages, config| async move {
            // Loaded fresh per instance and dropped when this future
            // completes, so instance 2 really starts from an empty hot cache.
            let model = qwen3_5_load_with_thread(&model_dir.to_string_lossy()).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}
