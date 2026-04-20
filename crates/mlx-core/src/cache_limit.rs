//! Auto-tuned Metal allocator cache limit + per-session memory hygiene.
//!
//! Python `mlx-lm` caps the Metal allocator's free-pool via
//! `mx.set_wired_limit(...)` at server startup and calls `mx.clear_cache()`
//! every 256 decode steps. We already mirror both of those: `WiredLimitContext`
//! (see `crates/mlx-core/src/stream.rs`) sets the wired limit at model load
//! and `synchronize_and_clear_cache()` drains the pool every 256 decode steps
//! inside each generative model. The piece that was still missing was an
//! explicit ceiling on how large the MLX allocator's free-pool may grow
//! between decode loops — on a 128GB M3 Max the default ceiling is the full
//! `max_recommended_working_set_size` (~96GB), so the pool slowly climbs to
//! that value and never drains on an idle process.
//!
//! ## Why a coordinator (not a one-shot function)
//!
//! `set_cache_limit` is a process-wide knob. An MLX-Node server can host
//! multiple generative models simultaneously (`ModelRegistry` has no upper
//! bound on concurrent `register()` calls). A naive
//! `apply_auto_cache_limit(model_bytes)` called from each model's `load()`
//! is a last-write-wins race: loading a small VLM after a big LM would
//! silently shrink the ceiling below the LM's working set.
//!
//! [`CacheLimitCoordinator`] tracks the live per-model deltas contributed
//! by each loaded model. Every call to [`CacheLimitCoordinator::register`]
//! returns a [`CacheLimitGuard`] tied to that model's lifetime. The
//! guard's `Drop` removes the entry and recomputes the ceiling, so
//! unloading one model reshapes the cap without ever leaving the
//! previously-capped value in place for a cold process — an empty
//! coordinator intentionally leaves the last-applied cap alone (nothing
//! to allocate anyway, cap costs nothing).
//!
//! ## Baseline choice: deterministic model-owned weight bytes
//!
//! Each caller passes its own delta computed as
//! `params.values().map(|a| a.nbytes()).sum::<usize>()` — the sum of
//! every weight array the model owns, in bytes. This value is:
//!
//!   - **Deterministic** — a pure function of the checkpoint and dtype
//!     layout, identical on every load.
//!   - **Model-local** — nothing the model does NOT own can contaminate
//!     the number, so there is no interaction with concurrent inference
//!     threads on a process-wide counter.
//!   - **Composable** — deltas sum naturally across models, so loading
//!     two models grows the cap to cover BOTH and unloading one shrinks
//!     it cleanly back to the survivor's footprint.
//!
//! An earlier iteration sampled `get_active_memory()` before/after the
//! load closure and used the delta. That was wrong: `get_active_memory()`
//! is a process-wide counter, so a concurrent inference thread
//! allocating between the before/after samples contaminated the delta
//! with memory that did not belong to the loading model, and the
//! corresponding unregister then shrunk the cap by the wrong amount. A
//! process-wide `LOAD_MUTEX` could serialize loads against each other
//! but could NOT serialize a load against live inference, so the race
//! was structurally unfixable without either blocking all inference
//! across load boundaries or abandoning the active-memory sample.
//! Deterministic weight bytes avoid the problem entirely — nothing in
//! the formula depends on observing process-global state, so there is
//! no race surface.
//!
//! ## Budget-based cap formula
//!
//! Earlier rounds computed the cap as `min(sum(weights) * 7/4, wired *
//! 3/5)`. That expression did NOT model the actual memory budget: the
//! `wired * 3/5` clamp scales only with the machine, not with how much
//! of wired the weights already occupy. Two real failure modes:
//!
//!   - **96 GB wired, 36 GB weights** → `96 * 0.6 = 57.6 GB` cap, peak
//!     memory ≈ `36 + 57.6 + ~10 driver` = 103 GB, which exceeds 96 GB
//!     wired and makes the whole system laggy.
//!   - **48 GB wired, 36 GB weights** → `48 * 0.6 = 28.8 GB` cap, peak
//!     ≈ `36 + 28.8 + ~10` = 75 GB → OOM, the model literally can't
//!     run.
//!
//! The new formula subtracts what is NOT the freelist from wired and
//! gives the remainder to MLX:
//!
//! ```text
//! cap = wired - weights - overhead - headroom      (if positive)
//!     = MIN_FREELIST_BYTES (1 GiB)                 (otherwise)
//! ```
//!
//! where
//!
//!   - **overhead** = `max(4 GiB, wired / 20)` — Metal driver state, MoE
//!     transpose cache, command buffer pool, kernel pipelines. Scales
//!     with system size with a floor for small-RAM hosts.
//!   - **headroom** = `max(4 GiB, wired / 10)` — reserved for macOS and
//!     other apps so the system stays responsive. Overridable via
//!     `MLX_GPU_HEADROOM_GB`.
//!
//! If weights + overhead + headroom already exceed wired (tight-fit
//! territory) we floor the freelist at 1 GiB so the allocator still
//! has something to reuse — MLX will churn but the model at least
//! runs.
//!
//! When wired is 0 (non-Metal machine or query failed) we fall back to
//! `weights * 3/2`, the same flavour of fixed multiplier as the old
//! formula's baseline term.
//!
//! ## Env overrides (precedence)
//!
//!   1. `MLX_CACHE_LIMIT_GB=N` — hard override, trumps everything. `=0`
//!      skips the call and retains the MLX default.
//!   2. `MLX_GPU_HEADROOM_GB=N` — tunes only the headroom term of the
//!      auto formula. Does NOT affect the overhead term.
//!   3. Otherwise: the budget formula above.
//!
//! ## Cache hygiene (no per-request RAII)
//!
//! An earlier iteration dropped a `ClearCacheOnDrop` guard inside every
//! session command handler. That is wrong on a multi-model server: the
//! allocator's free-pool is process-wide, so flushing after a request on
//! model A discards reusable blocks belonging to model B's next turn.
//! Between-turn draining now lives on the TS side (`@mlx-node/server`'s
//! idle sweeper — drains only when the whole process is idle for
//! `idleClearCacheMs`). The decode-loop `synchronize_and_clear_cache()`
//! fired every 256 steps is untouched.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use napi_derive::napi;
use tracing::info;

use crate::array::memory::{
    get_active_memory, get_cache_memory, get_peak_memory, set_cache_limit,
    synchronize_and_clear_cache,
};
use crate::stream::WiredLimitContext;

/// Name of the env var that overrides the auto-computed cache limit.
///
/// Value is parsed as a floating-point GB amount:
///   - `0`   → disable (do not call `set_cache_limit`, keep MLX defaults).
///   - `N>0` → explicit cap of `N * 1GiB` bytes.
///   - unset → use the auto formula.
pub const CACHE_LIMIT_ENV: &str = "MLX_CACHE_LIMIT_GB";

/// Name of the env var that tunes only the user-headroom term of the
/// auto formula. Parsed as a non-negative floating-point GiB amount;
/// invalid values are silently ignored and the default
/// (`max(4 GiB, wired / 10)`) is used instead. Does NOT affect the
/// driver-overhead term — use `MLX_CACHE_LIMIT_GB` for a hard override.
pub const GPU_HEADROOM_ENV: &str = "MLX_GPU_HEADROOM_GB";

const ONE_GIB: f64 = (1u64 << 30) as f64;
const GIB: u64 = 1u64 << 30;

/// Absolute floor on the freelist cap, in bytes. In tight-fit territory
/// (weights + overhead + headroom ≥ wired) we still hand the allocator
/// 1 GiB so it has something to reuse — MLX will churn but at least the
/// model runs instead of thrashing the allocator on every step.
const MIN_FREELIST_BYTES: u64 = GIB;

struct CoordState {
    next_id: u64,
    /// `guard_id -> weight_bytes`: per-model weight-byte totals
    /// captured by the caller as `sum(params.values().nbytes())`
    /// over every weight array the model owns. Summed (not max'd)
    /// so the cap tracks the true total working set across loaded
    /// models: unload subtracts cleanly and load adds cleanly.
    profiles: HashMap<u64, u64>,
    /// Most recent cap we actually pushed through `set_cache_limit`. Used
    /// so `recompute_locked` can short-circuit when the cap did not
    /// change — avoids log spam on every register/unregister.
    last_applied: Option<usize>,
}

/// Process-wide coordinator that owns the current MLX cache ceiling.
///
/// Register each loaded model via [`CacheLimitCoordinator::register`]; the
/// returned [`CacheLimitGuard`] unregisters on drop. All mutations are
/// serialized through a single `Mutex` — contention is low because
/// register/unregister happen once per model load/drop, not per request.
pub struct CacheLimitCoordinator {
    state: Mutex<CoordState>,
}

impl CacheLimitCoordinator {
    fn new() -> Self {
        Self {
            state: Mutex::new(CoordState {
                next_id: 1,
                profiles: HashMap::new(),
                last_applied: None,
            }),
        }
    }

    /// Register a model's weight-byte footprint and return an RAII
    /// guard that unregisters it on drop.
    ///
    /// `weight_bytes` should be the sum of `nbytes()` across every
    /// weight array the model owns (`params.values().map(|a|
    /// a.nbytes()).sum::<usize>()` as a `u64`). This is a
    /// deterministic value derived from the checkpoint and dtype
    /// layout — it does NOT depend on process-wide counters, so two
    /// loads on different threads cannot contaminate each other's
    /// delta regardless of interleaving with live inference.
    ///
    /// The global cap is recomputed synchronously before this
    /// returns, so the caller observes the post-register cap by the
    /// time the guard is in hand.
    pub fn register(&self, weight_bytes: u64) -> CacheLimitGuard {
        let id = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let id = state.next_id;
            state.next_id = state.next_id.saturating_add(1);
            state.profiles.insert(id, weight_bytes);
            info!(
                "[cache_limit] register model guard={} weights={:.2} GB (live_guards={})",
                id,
                weight_bytes as f64 / ONE_GIB,
                state.profiles.len(),
            );
            recompute_locked(&mut state);
            id
        };
        CacheLimitGuard { id }
    }

    fn unregister(&self, id: u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.profiles.remove(&id).is_some() {
            // Recompute after unregister. If the last model unloaded,
            // `recompute_locked` leaves the existing cap in place — a
            // cold process has nothing to allocate anyway.
            recompute_locked(&mut state);
        }
    }
}

/// RAII token returned from [`CacheLimitCoordinator::register`]. Dropping
/// it unregisters the delta and triggers a recompute so the cap shrinks
/// back down when a model unloads.
///
/// Each generative model wrapper (`Qwen3_5Model`, `Qwen3_5MoeModel`,
/// `Qwen3Model`, `Gemma4Model`, `Lfm2Model`, `VLModel`, `QianfanOCRModel`)
/// holds one of these as a field so its lifetime matches the native
/// model's lifetime. JS GC of the wrapper → `Drop` on the guard →
/// unregister.
pub struct CacheLimitGuard {
    id: u64,
}

impl Drop for CacheLimitGuard {
    fn drop(&mut self) {
        coordinator().unregister(self.id);
    }
}

/// Access the process-wide coordinator, initializing it on first use.
pub fn coordinator() -> &'static CacheLimitCoordinator {
    static INSTANCE: OnceLock<CacheLimitCoordinator> = OnceLock::new();
    INSTANCE.get_or_init(CacheLimitCoordinator::new)
}

fn recompute_locked(state: &mut CoordState) {
    // Env override takes absolute precedence and bypasses the baseline
    // tracking entirely. Behaviour preserved verbatim from the previous
    // one-shot implementation so existing deployments do not regress.
    if let Ok(raw) = std::env::var(CACHE_LIMIT_ENV) {
        let trimmed = raw.trim();
        match trimmed.parse::<f64>() {
            Ok(gib) if gib <= 0.0 => {
                // Log once per sticky state transition — first call with
                // env=0 logs; subsequent register/unregister calls with
                // the same env=0 sentinel are silent.
                if state.last_applied != Some(0) {
                    info!(
                        "[cache_limit] {}={} → skipping auto cache limit (MLX default retained)",
                        CACHE_LIMIT_ENV, trimmed
                    );
                }
                state.last_applied = Some(0);
                return;
            }
            Ok(gib) => {
                let bytes = (gib * ONE_GIB).round() as usize;
                if state.last_applied != Some(bytes) {
                    apply_limit(bytes, &format!("env {}={}", CACHE_LIMIT_ENV, trimmed));
                    state.last_applied = Some(bytes);
                }
                return;
            }
            Err(_) => {
                // Parse failure only logged once per distinct (apply, recompute)
                // cycle — fall through to auto formula below.
                info!(
                    "[cache_limit] Ignoring unparseable {}={:?}, using auto formula",
                    CACHE_LIMIT_ENV, raw
                );
            }
        }
    }

    // Empty coordinator → nothing to cap. Do NOT reset the last-applied
    // cap: the allocator state the prior cap was protecting is gone, so
    // the cap costs nothing; resetting just churns logs.
    if state.profiles.is_empty() {
        return;
    }
    // Sum (not max) across live weight-byte totals: each caller
    // registered its own per-model footprint, so summing gives the
    // true multi-model working-set baseline. `saturating_add` guards
    // against a measurement anomaly producing a huge bogus value
    // overflowing u64 when combined with others.
    let summed_weights: u64 = state
        .profiles
        .values()
        .copied()
        .fold(0u64, |acc, v| acc.saturating_add(v));
    if summed_weights == 0 {
        // All weight-byte totals were zero (unlikely — should only
        // happen in a synthetic test that registers a zero). Skip
        // rather than set a zero ceiling that would deadlock the
        // allocator.
        return;
    }

    let wired = WiredLimitContext::get_max_working_set_size() as u64;
    let limit = compute_cache_limit(summed_weights, wired);

    if limit == 0 {
        return;
    }

    let bytes = limit as usize;
    if state.last_applied == Some(bytes) {
        return;
    }

    // Build the `source` string so the operator can reconstruct the
    // full budget breakdown from logs. The env-override path emits its
    // own string at the top of `recompute_locked`.
    let source = if wired == 0 {
        format!(
            "auto (weights={:.1}GB, wired=0 → fallback cap={:.1}GB (weights × 1.5), live_guards={})",
            summed_weights as f64 / ONE_GIB,
            limit as f64 / ONE_GIB,
            state.profiles.len(),
        )
    } else {
        let overhead = estimate_metal_overhead(wired);
        let headroom = estimate_user_headroom(wired);
        format!(
            "auto (weights={:.1}GB, overhead={:.1}GB, headroom={:.1}GB, wired={:.1}GB → cap={:.1}GB, live_guards={})",
            summed_weights as f64 / ONE_GIB,
            overhead as f64 / ONE_GIB,
            headroom as f64 / ONE_GIB,
            wired as f64 / ONE_GIB,
            limit as f64 / ONE_GIB,
            state.profiles.len(),
        )
    };

    apply_limit(bytes, &source);
    state.last_applied = Some(bytes);
}

/// Estimate the Metal driver's own overhead footprint for the given
/// wired limit. Covers driver state, MoE weight-transpose caches,
/// command-buffer pool, kernel-pipeline state.
///
/// Scales at 5% of wired with a 4 GiB floor for small-RAM hosts where
/// even a small driver footprint matters.
fn estimate_metal_overhead(wired: u64) -> u64 {
    core::cmp::max(4 * GIB, wired / 20)
}

/// Estimate the memory that should stay reserved for macOS and other
/// user apps so the system remains responsive during inference.
///
/// Defaults to 10% of wired with a 4 GiB floor. If `MLX_GPU_HEADROOM_GB`
/// is set and parses as a non-negative finite float, its value wins
/// (in GiB). Non-parseable or negative values are ignored.
fn estimate_user_headroom(wired: u64) -> u64 {
    if let Ok(raw) = std::env::var(GPU_HEADROOM_ENV)
        && let Ok(gib) = raw.trim().parse::<f64>()
        && gib >= 0.0
        && gib.is_finite()
    {
        return (gib * GIB as f64).round() as u64;
    }
    core::cmp::max(4 * GIB, wired / 10)
}

/// Compute the freelist cap from total weight bytes and the Metal
/// wired limit, using the budget formula described at the top of this
/// module.
///
/// Contract:
///   - `wired == 0` → assume non-Metal or failed query, fall back to
///     `weights * 3/2`.
///   - `weights + overhead + headroom >= wired` → return
///     [`MIN_FREELIST_BYTES`] (tight-fit floor).
///   - otherwise → `wired - weights - overhead - headroom`, clamped to
///     at least [`MIN_FREELIST_BYTES`].
fn compute_cache_limit(weights: u64, wired: u64) -> u64 {
    if wired == 0 {
        return weights.saturating_mul(3) / 2;
    }
    let overhead = estimate_metal_overhead(wired);
    let headroom = estimate_user_headroom(wired);
    let reserved = weights.saturating_add(overhead).saturating_add(headroom);
    if wired <= reserved {
        return MIN_FREELIST_BYTES;
    }
    (wired - reserved).max(MIN_FREELIST_BYTES)
}

fn apply_limit(bytes: usize, source: &str) {
    let prev = set_cache_limit(bytes as f64);
    info!(
        "[cache_limit] cache pool cap set to {:.1} GB ({}); previous = {:.1} GB",
        bytes as f64 / ONE_GIB,
        source,
        prev / ONE_GIB,
    );
}

// ── Minimal JS-facing surface ──────────────────────────────────────
//
// We deliberately expose only two escape hatches to TypeScript:
//
//   - `clearCache()` — manual drain when callers know better than the
//     auto cadence (e.g. after a big prefill that consumed a lot of
//     scratch, or before a long idle period in a custom server). The
//     TS idle sweeper in `@mlx-node/server` calls this. Gated behind
//     an `__internal__` NAPI namespace so it does NOT land on the
//     root `require('@mlx-node/core')` object — user code has to
//     reach through `core.__internal__.clearCache` explicitly, which
//     makes the unsafe-stream caveat visible at the call site.
//   - `memoryStats()` — read-only snapshot for dashboards / debugging.
//     Stays on the root surface because it can't damage allocator
//     state.
//
// Everything else on `memory.rs` (synchronize, set_cache_limit,
// set_wired_limit, reset_peak_memory, heavy_cleanup) stays Rust-internal:
// the memory budget is owned by the native layer and manual overrides
// from JS are a footgun.

/// Snapshot of the MLX Metal allocator's memory state. All values are in
/// bytes and returned as `f64` to avoid forcing BigInt round-trips in JS.
#[napi(object, js_name = "MemoryStats")]
#[derive(Clone, Debug)]
pub struct MemoryStats {
    /// Actively-used memory (excludes the cached free-pool).
    pub active: f64,
    /// Peak memory usage since load / the last `resetPeakMemory`.
    pub peak: f64,
    /// Cache / free-pool memory currently held by the allocator.
    pub cache: f64,
    /// Metal `max_recommended_working_set_size` snapshot (0 on non-Metal).
    pub wired_limit: f64,
}

/// Drain the MLX allocator's free-pool.
///
/// @internal
///
/// This is a process-wide drain routed through MLX's default-stream
/// `mlx_synchronize()`, which does NOT wait on the custom generation
/// streams that the per-model threads run on. Calling this from user
/// code while a decode is in flight can race live Metal command buffers
/// and risk use-after-free. The only safe caller today is
/// `@mlx-node/server`'s idle sweeper, which only triggers after the
/// in-flight request counter has returned to zero.
///
/// Exposed under the `__internal__` NAPI namespace — reachable as
/// `require('@mlx-node/core').__internal__.clearCache()` and NOT on
/// the root `require('@mlx-node/core')` object. The namespace prefix
/// is a deliberate speed-bump that forces any caller to acknowledge
/// this is a private drain with custom-stream caveats; the root
/// surface stays clean of the footgun.
#[napi(namespace = "__internal__")]
pub fn clear_cache() {
    synchronize_and_clear_cache();
}

/// Return a snapshot of the MLX allocator's memory counters. Primarily
/// useful for dashboards and for debugging the `MLX_CACHE_LIMIT_GB`
/// override. Read-only — does not mutate allocator state.
#[napi]
pub fn memory_stats() -> MemoryStats {
    MemoryStats {
        active: get_active_memory(),
        peak: get_peak_memory(),
        cache: get_cache_memory(),
        wired_limit: WiredLimitContext::get_max_working_set_size() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that touch process-global env vars so one test
    /// never observes another's `MLX_GPU_HEADROOM_GB` setting.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that unsets the env var on drop. Any test that calls
    /// `std::env::set_var(GPU_HEADROOM_ENV, ...)` should wrap the call
    /// in one of these so a panic cannot leak the variable into the
    /// next test.
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: tests that invoke this serialize on `ENV_LOCK`,
            // so no other test is concurrently reading or writing
            // this var. Production code does not call `set_var` on
            // either of the cache-limit env vars.
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: see `EnvGuard::set`.
            unsafe {
                match self.prev.take() {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    const GB: u64 = 1u64 << 30;

    /// Clear any stale `MLX_GPU_HEADROOM_GB` before asserting on the
    /// auto formula. Must be called while holding `ENV_LOCK`.
    fn clear_headroom_env() {
        // SAFETY: callers hold `ENV_LOCK` so no concurrent access.
        unsafe {
            std::env::remove_var(GPU_HEADROOM_ENV);
        }
    }

    /// Tolerance helper for GiB-level assertions: integer rounding in
    /// the overhead/headroom derivations can shift the cap by a few
    /// bytes, so we allow 0.1 GiB of slack.
    fn approx_eq_gb(actual: u64, expected_gb: f64) {
        let actual_gb = actual as f64 / ONE_GIB;
        let diff = (actual_gb - expected_gb).abs();
        assert!(
            diff <= 0.1,
            "expected cap ≈ {expected_gb:.2} GB, got {actual_gb:.4} GB (diff {diff:.4})",
        );
    }

    // ── compute_cache_limit sanity cases ──────────────────────────

    #[test]
    fn case1_96gb_wired_36gb_weights() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = max(4, 96/20=4.8) = 4.8 GB
        // headroom = max(4, 96/10=9.6) = 9.6 GB
        // cap = 96 - 36 - 4.8 - 9.6 = 45.6 GB
        let cap = compute_cache_limit(36 * GB, 96 * GB);
        approx_eq_gb(cap, 45.6);
    }

    #[test]
    fn case2_48gb_wired_36gb_weights() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = max(4, 48/20=2.4) = 4 GB (floor)
        // headroom = max(4, 48/10=4.8) = 4.8 GB
        // cap = 48 - 36 - 4 - 4.8 = 3.2 GB
        let cap = compute_cache_limit(36 * GB, 48 * GB);
        approx_eq_gb(cap, 3.2);
    }

    #[test]
    fn case3_48gb_wired_42gb_weights_hits_floor() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = 4 GB, headroom = 4.8 GB
        // reserved = 42 + 4 + 4.8 = 50.8 GB > 48 GB wired → floor
        let cap = compute_cache_limit(42 * GB, 48 * GB);
        assert_eq!(cap, MIN_FREELIST_BYTES);
        approx_eq_gb(cap, 1.0);
    }

    #[test]
    fn case4_192gb_wired_36gb_weights() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = max(4, 192/20=9.6) = 9.6 GB
        // headroom = max(4, 192/10=19.2) = 19.2 GB
        // cap = 192 - 36 - 9.6 - 19.2 = 127.2 GB
        let cap = compute_cache_limit(36 * GB, 192 * GB);
        approx_eq_gb(cap, 127.2);
    }

    #[test]
    fn case5_192gb_wired_10gb_weights() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = 9.6 GB, headroom = 19.2 GB
        // cap = 192 - 10 - 9.6 - 19.2 = 153.2 GB
        let cap = compute_cache_limit(10 * GB, 192 * GB);
        approx_eq_gb(cap, 153.2);
    }

    #[test]
    fn wired_zero_falls_back_to_weights_times_one_and_a_half() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        let cap = compute_cache_limit(10 * GB, 0);
        // 10 * 3 / 2 = 15 GB
        approx_eq_gb(cap, 15.0);
    }

    // ── env override: MLX_GPU_HEADROOM_GB ─────────────────────────

    #[test]
    fn headroom_env_overrides_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "20");
        // overhead (96/20=4.8) = 4.8 GB, headroom forced to 20 GB
        // cap = 96 - 36 - 4.8 - 20 = 35.2 GB
        let cap = compute_cache_limit(36 * GB, 96 * GB);
        approx_eq_gb(cap, 35.2);
    }

    #[test]
    fn headroom_env_zero_is_honoured() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "0");
        // overhead = 4.8, headroom forced to 0
        // cap = 96 - 36 - 4.8 - 0 = 55.2 GB
        let cap = compute_cache_limit(36 * GB, 96 * GB);
        approx_eq_gb(cap, 55.2);
    }

    #[test]
    fn headroom_env_invalid_falls_back_to_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "not-a-number");
        // Invalid → default 10% applies → same as case 1.
        let cap = compute_cache_limit(36 * GB, 96 * GB);
        approx_eq_gb(cap, 45.6);
    }

    #[test]
    fn headroom_env_negative_is_ignored() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "-5");
        // Negative → rejected → default 10% applies.
        let cap = compute_cache_limit(36 * GB, 96 * GB);
        approx_eq_gb(cap, 45.6);
    }

    // ── overhead / headroom helper sanity ─────────────────────────

    #[test]
    fn overhead_floor_applies_on_small_systems() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // 16 / 20 = 0.8 GB → clamped up to 4 GB floor.
        assert_eq!(estimate_metal_overhead(16 * GB), 4 * GB);
    }

    #[test]
    fn overhead_scales_on_big_systems() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // 192 / 20 = 9.6 GB > 4 GB floor.
        assert_eq!(estimate_metal_overhead(192 * GB), 192 * GB / 20);
    }

    #[test]
    fn headroom_floor_applies_on_small_systems() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // 16 / 10 = 1.6 GB → clamped up to 4 GB floor.
        assert_eq!(estimate_user_headroom(16 * GB), 4 * GB);
    }

    #[test]
    fn headroom_scales_on_big_systems() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // 192 / 10 = 19.2 GB > 4 GB floor.
        assert_eq!(estimate_user_headroom(192 * GB), 192 * GB / 10);
    }
}
