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
//! ### Why 1.75× and not a tighter multiplier
//!
//! The deterministic weight-byte baseline does NOT include allocations
//! built lazily on the first prefill — the canonical example is the
//! MoE weight-transpose cache `g_weight_transposes_3d` built inside
//! `mlx_qwen35_moe_init_from_prefill`, not at load time. Compiled-graph
//! scratch buffers and first-prefill activations add more.
//!
//! The 1.75× multiplier is empirical slack to cover that post-load
//! scratch without requiring runtime measurements that race with
//! concurrent inference. Rough sizing: a Q8 35B MoE ≈ 35 GB of weights
//! → cap ≈ 61 GB; typical working set including transpose caches ≈
//! 40–50 GB; headroom > 10 GB. The cap is further clamped by
//! `wired * 3/5` so we never exceed 60% of the Metal wired limit.
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

const ONE_GIB: f64 = (1u64 << 30) as f64;

/// Baseline-bytes multiplier numerator (`7/4 = 1.75`).
///
/// Rationale: the summed per-model weight-byte total misses allocations
/// built lazily on the first prefill (e.g. the MoE weight-transpose
/// cache in `mlx_qwen35_moe_init_from_prefill`, compiled-graph scratch,
/// first-prefill activations). The 75% slack absorbs that post-load
/// growth without pushing the cap above `wired * 3/5`.
const BASELINE_MULT_NUM: u64 = 7;
const BASELINE_MULT_DEN: u64 = 4;

/// Wired-limit fraction (`3/5 = 0.6`). Headroom for OS, other GPU
/// consumers, and the allocator's own fragmentation slack.
const WIRED_FRAC_NUM: u64 = 3;
const WIRED_FRAC_DEN: u64 = 5;

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

    let wired = WiredLimitContext::get_max_working_set_size();
    let by_baseline = summed_weights.saturating_mul(BASELINE_MULT_NUM) / BASELINE_MULT_DEN;
    let by_wired = (wired as u64).saturating_mul(WIRED_FRAC_NUM) / WIRED_FRAC_DEN;

    // If wired is 0 (Metal unavailable / query failed) fall back to the
    // baseline term only. On a machine without Metal the setter is a
    // no-op but we still record `last_applied` for idempotence.
    let limit = if wired == 0 {
        by_baseline
    } else {
        by_baseline.min(by_wired)
    };

    if limit == 0 {
        return;
    }

    let bytes = limit as usize;
    if state.last_applied == Some(bytes) {
        return;
    }

    apply_limit(
        bytes,
        &format!(
            "auto (sum_weights={:.1}GB × {:.2}, wired={:.1}GB × {:.2}, live_guards={})",
            summed_weights as f64 / ONE_GIB,
            BASELINE_MULT_NUM as f64 / BASELINE_MULT_DEN as f64,
            wired as f64 / ONE_GIB,
            WIRED_FRAC_NUM as f64 / WIRED_FRAC_DEN as f64,
            state.profiles.len(),
        ),
    );
    state.last_applied = Some(bytes);
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
