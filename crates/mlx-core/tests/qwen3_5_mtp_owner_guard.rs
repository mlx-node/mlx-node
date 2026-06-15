//! MTP draft/commit graph ownership-guard logic test (dense + MoE).
//!
//! The dense `g_mtp_draft_decode_compiled` / `g_commit_graph_by_m` and the MoE
//! `g_moe_mtp_draft_decode_compiled` bake one model's MTP head weights into a
//! PROCESS-WIDE compiled tape. P5 step-2 slotted the per-model VERIFY graph, but
//! those draft/commit graphs stayed process-wide. When two MTP models alternate
//! in one process, the second model's draft/commit would replay the first's
//! baked tape and draft against the wrong weights — output stays correct (the
//! per-model verify graph + accept test reject a bad draft) but speculative
//! acceptance tanks.
//!
//! The ownership guard (`qwen35_mtp_ensure_owner` / `moe_mtp_ensure_owner`)
//! records which model's weights are baked into the process-wide graphs and
//! nulls them on a model switch so the lazy getter re-traces against the active
//! registry. This test drives that LOGIC directly through test-only FFI hooks —
//! no real MTP checkpoint, no full init — by setting the shared active model id
//! (`mlx_set_model_id`) and observing the owner id + whether the baked graph was
//! invalidated.
//!
//! Two key properties proven:
//!   1. SWITCH invalidates: stamping a sentinel graph for model A, then making B
//!      active and running ensure_owner, must null the graph AND set owner = B.
//!   2. SAME-MODEL NO-OP: with the id unchanged, ensure_owner must NOT null a
//!      sentinel graph (this is the byte/perf-identical single-model invariant).
//!
//! No Metal ops run (ensure_owner only nulls std::functions + reads an atomic),
//! so the test is host-independent. Lives in `mlx-core` (not `mlx-paged-attn`)
//! because mlx-core's lib references `mlx_qwen35_mtp_draft_compiled` /
//! `mlx_qwen35_moe_mtp_draft_compiled`, which pull the MTP objects (and thus
//! these test-only symbols) into the test binary's static link. Single combined
//! `#[test]` so the process-global owner ids / active id are never touched
//! concurrently.

#![cfg(target_os = "macos")]

const MODEL_A: u64 = 0xA1;
const MODEL_B: u64 = 0xB2;

#[test]
fn mtp_owner_guard_invalidates_on_switch_and_noops_on_same_model() {
    unsafe {
        // ---------------- DENSE ----------------

        // Adopt A as the owner: set the active id to A, stamp a non-null
        // sentinel into the baked graphs, then run the guard.
        mlx_sys::mlx_set_model_id(MODEL_A);
        mlx_sys::mlx_qwen35_mtp_test_set_graph_sentinels();
        mlx_sys::mlx_qwen35_mtp_test_ensure_owner();
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_owner_id(),
            MODEL_A,
            "dense: ensure_owner must adopt the active model id A"
        );
        // That first guard saw owner change (initial -> A) and nulled the
        // sentinels, so restamp to set up the no-op check below.
        mlx_sys::mlx_qwen35_mtp_test_set_graph_sentinels();
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_draft_graph_is_null(),
            0,
            "dense: sentinel draft graph must be non-null before the no-op check"
        );

        // SAME-MODEL NO-OP: id unchanged (still A) → ensure_owner must NOT null
        // the sentinel graphs. This is the single-model byte/perf invariant.
        mlx_sys::mlx_qwen35_mtp_test_ensure_owner();
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_owner_id(),
            MODEL_A,
            "dense: owner must stay A when the active id did not change"
        );
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_draft_graph_is_null(),
            0,
            "dense: same-model ensure_owner nulled the draft graph — NOT a no-op"
        );
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_commit_graphs_all_null(),
            0,
            "dense: same-model ensure_owner nulled the commit graphs — NOT a no-op"
        );

        // SWITCH INVALIDATES: make B active. The sentinel graphs still belong to
        // A; ensure_owner must null BOTH the draft and the commit graphs and set
        // the owner to B.
        mlx_sys::mlx_set_model_id(MODEL_B);
        mlx_sys::mlx_qwen35_mtp_test_ensure_owner();
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_owner_id(),
            MODEL_B,
            "dense: ensure_owner must adopt B after the active id switched to B"
        );
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_draft_graph_is_null(),
            1,
            "dense: switching to B must invalidate A's baked draft graph"
        );
        assert_eq!(
            mlx_sys::mlx_qwen35_mtp_test_commit_graphs_all_null(),
            1,
            "dense: switching to B must invalidate A's baked commit graphs"
        );

        // ---------------- MoE ----------------
        // (MoE has only a draft graph — no commit FFI.)

        mlx_sys::mlx_set_model_id(MODEL_A);
        mlx_sys::mlx_qwen35_moe_mtp_test_set_graph_sentinel();
        mlx_sys::mlx_qwen35_moe_mtp_test_ensure_owner();
        assert_eq!(
            mlx_sys::mlx_qwen35_moe_mtp_test_owner_id(),
            MODEL_A,
            "moe: ensure_owner must adopt the active model id A"
        );
        mlx_sys::mlx_qwen35_moe_mtp_test_set_graph_sentinel();
        assert_eq!(
            mlx_sys::mlx_qwen35_moe_mtp_test_draft_graph_is_null(),
            0,
            "moe: sentinel draft graph must be non-null before the no-op check"
        );

        // SAME-MODEL NO-OP.
        mlx_sys::mlx_qwen35_moe_mtp_test_ensure_owner();
        assert_eq!(
            mlx_sys::mlx_qwen35_moe_mtp_test_owner_id(),
            MODEL_A,
            "moe: owner must stay A when the active id did not change"
        );
        assert_eq!(
            mlx_sys::mlx_qwen35_moe_mtp_test_draft_graph_is_null(),
            0,
            "moe: same-model ensure_owner nulled the draft graph — NOT a no-op"
        );

        // SWITCH INVALIDATES.
        mlx_sys::mlx_set_model_id(MODEL_B);
        mlx_sys::mlx_qwen35_moe_mtp_test_ensure_owner();
        assert_eq!(
            mlx_sys::mlx_qwen35_moe_mtp_test_owner_id(),
            MODEL_B,
            "moe: ensure_owner must adopt B after the active id switched to B"
        );
        assert_eq!(
            mlx_sys::mlx_qwen35_moe_mtp_test_draft_graph_is_null(),
            1,
            "moe: switching to B must invalidate A's baked draft graph"
        );

        // Cleanup: drop the active id back to "no model" so a later test in the
        // same process binary starts from a clean active id.
        mlx_sys::mlx_set_model_id(0);
    }
}
