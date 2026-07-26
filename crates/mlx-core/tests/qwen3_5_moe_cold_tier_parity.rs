//! Real-weights cold-tier restart-parity gate for Qwen3.5 MoE (hybrid GDN).
//!
//! Same three-instance scenario as the qwen3 gate — see
//! [`cold_tier_parity_harness`] — and the SAME hybrid meaning a pass carries for
//! dense qwen3_5: the MoE variant sizes its paged pool over the FULL-ATTENTION
//! layers only, and keeps its GDN (gated delta-net) recurrent state in a
//! per-layer `ArraysCache` (conv + recurrent) OUTSIDE the pool. That state is
//! byte-identical to the dense family's — same shapes, same dtype, same
//! layer mapping (`Qwen3_5MoeConfig::to_dense_config` projects the linear
//! geometry), so both share ONE GDN sidecar codec
//! (`crate::models::qwen3_5::gdn_sidecar`).
//!
//! # Why a pass here is GDN-sidecar evidence, not just KV evidence
//!
//! With a `ColdSidecarPolicy` installed, `ColdTierWalk::restore_extend` restores
//! NOTHING unless a validated GDN sidecar backs the boundary
//! (`deepest_backed_boundary` -> `None` => `ColdRestore::miss()`). Instance 2 is
//! a freshly loaded model, so its in-process prefix cache is empty and a hot hit
//! is impossible. Any non-zero `cached_tokens` it reports can therefore only have
//! come through the reconcile-down walk — a GDN state sidecar was found on disk,
//! decoded, and validated against this checkpoint's geometry.
//!
//! Found + read + checksummed + layout-validated is NOT the same as INSTALLED:
//! `install_moe_gdn_cold_sidecar`'s `Ok(false)` arms all fall through to a full
//! O(prefix) GDN replay whose output, `cached_tokens`, `hits` and `corruptions`
//! are indistinguishable from a healthy run. `expecting_sidecar_install()`
//! asserts the `installed` counter advanced during instance 2 alone, and only
//! with it set does a pass here mean "the tier restored the recurrent half too".
//! See the dense sibling for the long version.
//!
//! # Why warm-up turns
//!
//! The GDN sidecar may sit at any rung of the prefill's checkpoint ladder
//! (`gdn_prefill_checkpoint_boundaries`): the deepest rung is
//! `gdn_checkpoint_target` — the largest full block strictly before the end of
//! the prompt — and each shallower rung is a block-aligned quarter of the one
//! above it. But `ColdTierWalk::capture_chain` stops at the first block the
//! bounded writer queue refuses, so one turn persists only the first handful of
//! K/V blocks. A sidecar is only WRITTEN at the deepest rung the persisted K/V
//! chain already covers (`cold_captured_blocks`), so a few warm-up turns deepen
//! the chain until a rung qualifies. Blocks already on disk are skipped without
//! re-enqueueing, so the frontier advances every turn.
//!
//! # Why the capture and restore prompts differ
//!
//! Same reason as the dense gate, and the same shared fixture
//! (`harness::ladder_capture_prompt` / `ladder_restore_prompt`, built by one
//! builder so the two families cannot drift into two differently-sized claims).
//! A gate whose three instances all run a ~90-token prompt cannot see a ladder:
//! the ladder there is `[16, 80]` and one turn's chain reach (~8 blocks = 128
//! tokens) already covers the deepest rung, so the restore always anchors at
//! the prompt's end and a ladder collapsed to one endpoint boundary leaves the
//! gate green. With the ladder fixture the deepest rung is tens of blocks in,
//! and the point where the two prompts diverge caps `kv_chain_upper_bound` far
//! below it — so the restore MUST reconcile onto a shallower rung, which is
//! what the harness's assertion 1b checks. It kills a one-rung ladder; it does
//! not pin the rung count at four, nor the ratio — both of those are pinned by
//! exact rung values in
//! `qwen3_5::paged_forward::gdn_checkpoint_tests::ladder_rungs_are_quarters_of_the_one_above`,
//! model-free and in milliseconds. See the dense gate's module doc.
//!
//! Observed on `Qwen3.6-35b-a3b-UD-Q2_K_XL-mlx` (40 layers, `full_attention_interval`
//! 4, `head_dim` 256, no MTP head), 105 s wall over 5 fresh loads:
//!
//! ```text
//! capture prompt 1259 tok -> ladder [16, 64, 304, 1248]   (deepest = 78 blocks)
//! warm-up 1 cached=0     warm-up 2 cached=64     instance 1 cached=304
//! instance 2 (restart)   cached=304   <-- rung 3 of 4, strictly below 1248
//! sidecar telemetry      boundary_skips=0 already_persisted=2 enqueued=0
//! cold stats             hits=42 misses=0 corruptions=0
//! ```
//!
//! Rung for rung the same shape the dense 0.8B gate reports, because both
//! families tokenize this fixture to the same 1259 tokens — this file used to
//! hedge that "the MoE tokenizer will differ", and it does not. Nothing here is
//! a constant either way: every assertion reads the runtime `prompt_tokens`.
//!
//! `already_persisted=2` with `enqueued=0` is the healthy steady state, not a
//! miss — a warm-up turn wrote rung 304 and both measured turns re-selected the
//! same chain and deduped. Under a ladder collapsed to one rung the same three
//! lines instead read `cached=0` everywhere, `boundary_skips=2
//! already_persisted=0`, and `restore anchored at 0`, and the harness's
//! assertion 1 fires.
//!
//! Gated on `MLX_TEST_MODEL_PATH`. The tier manager is a process-global
//! `OnceLock`, so this must be the only thing in the process touching it —
//! hence `#[ignore]` plus `--test-threads=1`. This is a large MoE checkpoint, so
//! allow generous time.
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//!     MLX_TEST_MODEL_PATH=~/.mlx-node/models/Qwen3.6-35b-a3b-UD-Q2_K_XL-mlx \
//!     cargo test -p mlx-core --test qwen3_5_moe_cold_tier_parity \
//!     -- --ignored --test-threads=1 --nocapture
//! ```

mod cold_tier_parity_harness;

use cold_tier_parity_harness as harness;
use mlx_core::models::qwen3_5_moe::persistence::load_with_thread as qwen3_5_moe_load_with_thread;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 MoE checkpoint; run with --test-threads=1"]
async fn qwen3_5_moe_cold_tier_restart_parity() {
    harness::run_restart_parity(
        harness::ColdTierParitySpec::new("qwen3_5_moe")
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
            // See the dense sibling: without this the gate cannot tell an
            // installed GDN sidecar from one read, validated and thrown away.
            .expecting_sidecar_install(),
        |model_dir, messages, config| async move {
            // Loaded fresh per instance and dropped when this future
            // completes, so instance 2 really starts from an empty hot cache.
            let model = qwen3_5_moe_load_with_thread(&model_dir.to_string_lossy()).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}
