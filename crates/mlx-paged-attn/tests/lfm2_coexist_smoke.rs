//! Coexistence smoke test for the LFM2 / LFM2-MoE per-model compiled slot.
//!
//! Proves that two lfm2 models (distinct `model_id`) each keep their OWN
//! compiled state resident at the same time — the property that replaces the
//! old single-`g_active_model_id` gate (where loading a 2nd compiled model
//! demoted the 1st to the eager path). Without the per-model `Lfm2Slot` map, the
//! two models would share the one set of file-scope working-register globals and
//! clobber each other. Mirrors `qwen3_5_coexist_smoke.rs` /
//! `qwen3_5_moe_coexist_smoke.rs`.
//!
//! The test drives the slot machinery directly through the FFI — no real model
//! weights and no full forward — using the `_test_*_paged_linear_slot` hooks to
//! stamp a sentinel into each model's slice of the dual-purpose
//! `g_lfm2_paged_k_pools` (the per-layer attn-K-pool / conv-state vector):
//!
//!   activate(A) → force caches → write slot0 = 42
//!   activate(B) → force caches → write slot0 = 99   (parks A into slot[A])
//!   activate(A) → read slot0 MUST be 42             (A survived B; the proof)
//!   activate(B) → read slot0 MUST be 99             (B survived A)
//!   erase(A) while B active → B's working register MUST be untouched
//!
//! Single `#[test]` (own test binary) so the process-global slot map / active-id
//! are never touched concurrently by another test thread.

#![cfg(target_os = "macos")]

const NUM_LAYERS: i32 = 4;
const FULL_ATTN_INTERVAL: i32 = 4;
const MODEL_A: u64 = 0xA1;
const MODEL_B: u64 = 0xB2;

/// 42 and 99 are integers < 256, so they round-trip through bf16 exactly — the
/// slot stores bf16 and the reader returns its f32 cast, so equality is exact.
const SENTINEL_A: f32 = 42.0;
const SENTINEL_B: f32 = 99.0;

fn metal_available() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

unsafe extern "C" {
    fn mlx_lfm2_activate_model(model_id: u64);
    fn mlx_lfm2_model_has_weights(model_id: u64) -> i32;
    fn mlx_lfm2_slot_erase(model_id: u64);
    fn mlx_lfm2_compiled_test_force_paged_linear_caches(
        num_layers: i32,
        full_attention_interval: i32,
    );
    fn mlx_lfm2_compiled_test_read_paged_linear_slot(slot_idx: i32) -> f32;
    fn mlx_lfm2_compiled_test_write_paged_linear_slot(slot_idx: i32, value: f32);
    fn mlx_lfm2_paged_reset();
    fn mlx_clear_weights(model_id: u64);
}

#[test]
fn lfm2_compiled_slots_isolate_two_models() {
    if !metal_available() {
        eprintln!(
            "skipping lfm2_compiled_slots_isolate_two_models: Metal unavailable on this host"
        );
        return;
    }

    unsafe {
        // Clean slate: drop any slots a prior run in this process left behind
        // and clear the paged working register.
        mlx_lfm2_slot_erase(MODEL_A);
        mlx_lfm2_slot_erase(MODEL_B);
        mlx_lfm2_paged_reset();

        // Model A becomes the working register; stamp its slot-0 sentinel.
        mlx_lfm2_activate_model(MODEL_A);
        mlx_lfm2_compiled_test_force_paged_linear_caches(NUM_LAYERS, FULL_ATTN_INTERVAL);
        mlx_lfm2_compiled_test_write_paged_linear_slot(0, SENTINEL_A);
        assert_eq!(
            mlx_lfm2_compiled_test_read_paged_linear_slot(0),
            SENTINEL_A,
            "model A sentinel must read back from its active working register"
        );

        // Activating B parks A into slot[A] and swaps B's (fresh) slot in.
        mlx_lfm2_activate_model(MODEL_B);
        mlx_lfm2_compiled_test_force_paged_linear_caches(NUM_LAYERS, FULL_ATTN_INTERVAL);
        mlx_lfm2_compiled_test_write_paged_linear_slot(0, SENTINEL_B);
        assert_eq!(
            mlx_lfm2_compiled_test_read_paged_linear_slot(0),
            SENTINEL_B,
            "model B sentinel must read back from its active working register"
        );

        // THE PROOF: A's slot survived B's whole turn (not clobbered to 99).
        mlx_lfm2_activate_model(MODEL_A);
        assert_eq!(
            mlx_lfm2_compiled_test_read_paged_linear_slot(0),
            SENTINEL_A,
            "model A compiled slot was clobbered by model B — coexistence broken"
        );

        // And B still independently holds its own sentinel.
        mlx_lfm2_activate_model(MODEL_B);
        assert_eq!(
            mlx_lfm2_compiled_test_read_paged_linear_slot(0),
            SENTINEL_B,
            "model B compiled slot was clobbered by model A — coexistence broken"
        );

        // Neither model registered weights, so the per-model compiled-capability
        // gate is false for both (the gate that replaced the active-id compare).
        assert_eq!(
            mlx_lfm2_model_has_weights(MODEL_A),
            0,
            "has_weights must be false for an unregistered model"
        );
        assert_eq!(mlx_lfm2_model_has_weights(MODEL_B), 0);

        // Erasing A's slot while B is the active model must not disturb B's
        // working register (erase only clears the register when it erases the
        // ACTIVE model).
        mlx_lfm2_slot_erase(MODEL_A);
        assert_eq!(
            mlx_lfm2_compiled_test_read_paged_linear_slot(0),
            SENTINEL_B,
            "erasing model A's slot disturbed the active model B working register"
        );

        // SAME-MODEL_ID RE-REGISTRATION GUARD. A's slot was just erased while A
        // was PARKED — exactly what the lfm2 loader now does (via
        // `mlx_lfm2_slot_erase(model_id)`) for the model it is (re-)registering.
        // For lfm2 the SHARED compile-epoch bump already forces a parked model's
        // slotted closure to rebuild on its next activate; the slot-erase
        // additionally drops the model's stale NON-closure state (its forced
        // `g_lfm2_paged_k_pools`) and `compile_erase`s its per-epoch fun_ids.
        // Re-activating A must therefore yield a FRESH, empty slot (its forced
        // caches are gone → out-of-range read → NaN), proving the stale
        // SENTINEL_A did NOT survive the erase. A real same-id reload then
        // re-seeds + re-traces against the newly stored weights instead of
        // swapping the old register back in.
        mlx_lfm2_activate_model(MODEL_A);
        assert!(
            mlx_lfm2_compiled_test_read_paged_linear_slot(0).is_nan(),
            "re-activating an erased (re-registered) model resurrected stale slot state"
        );

        // Cleanup: erase both models' slots (A is active now, B is parked) +
        // reset the working register.
        mlx_lfm2_slot_erase(MODEL_A);
        mlx_lfm2_slot_erase(MODEL_B);
        mlx_lfm2_paged_reset();
        mlx_clear_weights(0);
    }
}
