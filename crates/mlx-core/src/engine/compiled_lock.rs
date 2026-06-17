//! Process-wide synchronization for the compiled C++ forward paths.
//!
//! The compiled paths share process-global C++ state (the `g_weights()` map,
//! `g_active_model_id`, per-family compiled decode caches), so every family
//! that drives them must serialize on these one-per-process locks. Moved here
//! from `models::qwen3_5::model` so the contract lives in family-neutral code.

use std::sync::atomic::AtomicU64;

/// Monotonically incrementing counter for assigning unique model IDs across
/// EVERY family that drives the shared C++ compiled-weight registry. The C++
/// `g_weights()` map is process-global and keyed by model id, so ids must be
/// globally unique across all variants (Qwen3.5 dense/MoE, LFM2, …). Lives in
/// family-neutral code so the surviving compiled paths keep a single id space.
pub(crate) static QWEN35_MODEL_ID_COUNTER: AtomicU64 = AtomicU64::new(1); // 0 = no model

/// Block size hard-coded into the compiled C++ paged graphs (the
/// `paged_kv_write` / `paged_attention` kernel calls trace against
/// block_size=16). Adapters support configurable block sizes, but the
/// compile-branch selectors gate `cpp_session_ready` on this equality and fall
/// back to the pure-Rust paged path when the adapter is configured otherwise.
/// Family-neutral so every compiled paged path shares the one contract value.
pub(crate) const CPP_PAGED_REQUIRED_BLOCK_SIZE: u32 = 16;

/// RwLock protecting the C++ global weight map against concurrent mutation.
/// Write-locked during weight registration (model load), read-locked during
/// compiled inference. This prevents a concurrent model load from swapping
/// weights underneath an in-flight compiled decode, and eliminates the TOCTOU
/// between has_weight() / get_weight() in linear_proj().
pub(crate) static COMPILED_WEIGHTS_RWLOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Process-wide mutex serializing the compiled forward LIFECYCLE (per-turn
/// init / decode / reset) across model instances AND model families.
///
/// Within a single model instance the dedicated model thread serializes calls,
/// but distinct models run on distinct OS threads (one per model, see
/// `model_thread.rs`), so a qwen3.5 compiled decode and an lfm2 compiled decode
/// genuinely run in parallel. They collide on the SAME process-global C++
/// globals: the `g_weights()` map (read by the NOT id-aware `get_weight` /
/// `get_weight_t`), the shared `g_active_model_id` atom, and each family's
/// compiled decode state (`g_*_caches` / `g_*_offset_int`).
///
/// `pub(crate)` and family-agnostic by design: EVERY model that drives a
/// compiled path over the shared registry (qwen3.5 dense + MoE, lfm2, …) MUST
/// serialize its compiled lifecycle on THIS one instance — never a private
/// per-family mutex, which would provide zero mutual exclusion against an
/// in-flight compiled decode from another family.
pub(crate) static COMPILED_LIFECYCLE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the [`COMPILED_WEIGHTS_RWLOCK`] write lock, recovering from poison.
///
/// Used by the surviving compiled families' weight registration/clear paths
/// (e.g. lfm2 `persistence::register_weights_with_cpp`). A panic during a
/// prior registration/clear while holding this write lock poisons the
/// rwlock; a bare `.write().unwrap()` would then panic every subsequent
/// model load.
///
/// Poison recovery is sound here because the lock guards no Rust data (unit
/// payload): the protected state is the process-global C++ weight map, and
/// `mlx_clear_weights()` — the FIRST step of every registration — resets
/// `g_active_model_id` to 0, so a torn registration never publishes a model
/// id. A recovered writer immediately re-runs the FULL registration (clear →
/// store → publish id last), overwriting any torn state the poisoning panic
/// left behind. Compiled inference acquires the read lock inline with the
/// same `unwrap_or_else(|e| e.into_inner())` policy (lfm2's compiled path),
/// and re-validates `mlx_*_get_model_id() == self.model_id` under the guard,
/// falling back to the eager Rust forward when the check fails. Mirrors the
/// established policy on [`COMPILED_LIFECYCLE_MUTEX`].
pub(crate) fn compiled_weights_write() -> std::sync::RwLockWriteGuard<'static, ()> {
    COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Codex regression: a panic while HOLDING the weights write lock (a
    /// torn registration/clear) poisons the rwlock. The surviving
    /// compiled families' registration path acquires the write lock
    /// through [`compiled_weights_write`]; this must keep succeeding so a
    /// subsequent model load can re-register cleanly instead of panicking
    /// the model thread.
    ///
    /// NOTE: the poison flag deliberately stays set after recovery — every
    /// production acquisition goes through the recovering helper, so the
    /// sticky flag is harmless (also makes this test order-independent).
    #[test]
    fn compiled_weights_lock_recovers_from_poison() {
        // Poison the rwlock: panic on a scoped thread while holding the
        // write guard.
        let _ = std::thread::spawn(|| {
            let _guard = COMPILED_WEIGHTS_RWLOCK
                .write()
                .unwrap_or_else(|e| e.into_inner());
            panic!("poison COMPILED_WEIGHTS_RWLOCK for the recovery test");
        })
        .join();
        assert!(
            COMPILED_WEIGHTS_RWLOCK.is_poisoned(),
            "test setup failed to poison COMPILED_WEIGHTS_RWLOCK"
        );

        // The registration path must not panic.
        {
            let _write = compiled_weights_write();
        }
        // And a second registration after recovery still works (flag is
        // sticky).
        let _write_again = compiled_weights_write();
    }
}
