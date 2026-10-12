//! Auto-tuned Metal allocator cache limit + per-session memory hygiene.
//!
//! Python `mlx-lm` caps the Metal allocator's free-pool via
//! `mx.set_wired_limit(...)` at server startup and calls `mx.clear_cache()`
//! every 256 decode steps. We already mirror both of those: `WiredLimitContext`
//! (see `crates/mlx-core/src/stream.rs`) sets the wired limit at model load
//! and the FLAT `DecodeStep::maintain_cache` default (`crates/mlx-core/src/
//! engine/backend.rs`) drains the pool via `clear_cache()` every 256 decode
//! steps inside each generative model. The piece that was still missing was an
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
//! coordinator leaves the last automatic cap alone. A temporary model ceiling
//! instead restores the prior cap when no independent policy remains.
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
//! cap = wired - weights - paged pools - overhead - headroom (if positive)
//!     = MIN_FREELIST_BYTES (1 GiB)                      (otherwise)
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
//! Private paged-KV pools are registered separately because their Metal
//! buffers are invisible to MLX's allocator counters. If weights + pools +
//! overhead + headroom already exceed wired (tight-fit
//! territory) we floor the freelist at 1 GiB so the allocator still
//! has something to reuse — MLX will churn but the model at least
//! runs.
//!
//! When wired is 0 (non-Metal machine or query failed) we fall back to
//! `max(weights * 3/2 - paged pools, 1 GiB)`, the same flavour of fixed
//! multiplier as the old formula's baseline term without double-budgeting
//! a known private pool.
//!
//! ## Env overrides (precedence)
//!
//!   1. `MLX_CACHE_LIMIT_GB=N` — overrides the automatic budget. `=0`
//!      skips automatic policy. Explicit model ceilings still apply.
//!   2. `MLX_GPU_HEADROOM_GB=N` — tunes only the headroom term of the
//!      auto formula. Does NOT affect the overhead term.
//!   3. Otherwise: the budget formula above.
//!
//! Models can additionally register a lifetime-scoped free-pool ceiling via
//! [`CacheLimitCoordinator::register_with_cache_limit`]. The smallest live
//! ceiling constrains the process policy; this is not a per-model memory pool.
//!
//! ## Cache hygiene (no per-request RAII)
//!
//! An earlier iteration dropped a `ClearCacheOnDrop` guard inside every
//! session command handler. That is wrong on a multi-model server: the
//! allocator's free-pool is process-wide, so flushing after a request on
//! model A discards reusable blocks belonging to model B's next turn.
//! Between-turn draining now lives on the TS side (`@mlx-node/server`'s
//! idle sweeper — drains only when the whole process is idle for
//! `idleClearCacheMs`). The decode-loop `clear_cache()` fired every 256
//! steps is untouched.
//!
//! ## Decode-time ceiling (turn-scoped, see [`decode_cache_limit`])
//!
//! The load-time cap above is a *safety* ceiling (tens of GB on a big
//! host). A speculative decode loop needs nothing like that: each verify
//! cycle allocates a few dozen MiB of short-lived outputs and frees them
//! before the next cycle, so the free-pool only has to be large enough to
//! hand the next cycle its buffers back. Anything beyond that is resident
//! memory the process holds for no benefit until the next 256-token
//! `clear_cache()`. [`CacheLimitCoordinator::push_decode_limit`] lowers
//! the ceiling for the lifetime of a guard; the effective cap is
//! `min(load-time cap, min(model ceilings), min(active decode caps))`, recomputed on every
//! push/pop and on every model register/unregister, so a concurrent load
//! on another thread can never be clobbered by a stale "restore previous
//! value" write. An explicit `MLX_CACHE_LIMIT_GB` pin still trumps the
//! decode cap; explicit model ceilings still constrain that pin.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use napi_derive::napi;
use tracing::{info, warn};

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
    /// `guard_id -> profile`: per-model weight-byte totals and optional ceilings
    /// captured by the caller as `sum(params.values().nbytes())`
    /// over every weight array the model owns. Summed (not max'd)
    /// so the cap tracks the true total working set across loaded
    /// models: unload subtracts cleanly and load adds cleanly.
    profiles: HashMap<u64, ModelCacheProfile>,
    /// Private paged-KV pools allocate outside MLX's freelist counters but
    /// consume the same unified-memory working-set budget.
    pools: HashMap<u64, u64>,
    /// Turn-scoped ceilings owned by active decode loops.
    decode_limits: HashMap<u64, u64>,
    limit: AppliedCacheLimit,
}

struct ModelCacheProfile {
    weight_bytes: u64,
    cache_limit: Option<usize>,
}

impl CoordState {
    fn cache_ceiling(&self) -> Option<usize> {
        self.profiles.values().filter_map(|p| p.cache_limit).min()
    }

    fn scoped_cache_ceiling(&self, allow_decode_limit: bool) -> Option<usize> {
        let decode = allow_decode_limit
            .then(|| self.decode_limits.values().copied().min())
            .flatten()
            .map(|cap| usize::try_from(cap).unwrap_or(usize::MAX));
        self.cache_ceiling().into_iter().chain(decode).min()
    }
}

/// Separates an applied cap from an unmanaged policy. Retain the prior cap
/// while model or decode ceilings are live so removing the last one can restore
/// it even when there is no automatic/global policy to recompute.
#[derive(Default)]
struct AppliedCacheLimit {
    last_applied: Option<usize>,
    before_scoped_limits: Option<usize>,
}

impl AppliedCacheLimit {
    fn update(
        &mut self,
        policy: Option<usize>,
        ceiling: Option<usize>,
        set: impl FnOnce(usize) -> Option<usize>,
    ) {
        let target = match (policy, ceiling) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => self.before_scoped_limits,
        };
        let Some(target) = target else { return };
        let previous = if self.last_applied == Some(target) {
            target
        } else {
            let Some(previous) = set(target) else { return };
            self.last_applied = Some(target);
            previous
        };
        if ceiling.is_some() {
            self.before_scoped_limits.get_or_insert(previous);
        } else {
            self.before_scoped_limits = None;
        }
    }
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
                pools: HashMap::new(),
                decode_limits: HashMap::new(),
                limit: AppliedCacheLimit::default(),
            }),
        }
    }

    /// Lower the process-wide ceiling to at most `cap_bytes` for the
    /// lifetime of the returned guard (a running decode loop). Multiple
    /// live guards compose by `min`; the load-time cap is never exceeded.
    /// A `cap_bytes` of 0 is ignored (treated as "no decode cap") because a
    /// zero ceiling would make every `free` release its buffer immediately.
    pub fn push_decode_limit(&self, cap_bytes: u64) -> DecodeCacheLimitGuard {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let id = state.next_id;
        state.next_id = state.next_id.saturating_add(1);
        if cap_bytes > 0 {
            state.decode_limits.insert(id, cap_bytes);
            recompute_locked(&mut state);
        }
        DecodeCacheLimitGuard { id }
    }

    fn pop_decode_limit(&self, id: u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.decode_limits.remove(&id).is_some() {
            recompute_locked(&mut state);
        }
    }

    /// Smallest live decode cap, if any. Test/diagnostic accessor.
    #[cfg(test)]
    fn active_decode_limit(&self) -> Option<u64> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.decode_limits.values().copied().min()
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
        self.register_with_cache_limit(weight_bytes, None)
    }

    /// Register weights and an optional ceiling on the shared free-buffer pool.
    /// Live ceilings compose by minimum with the automatic/global policy.
    /// `None` adds no constraint; `Some(0)` disables free-buffer retention.
    /// Dropping the guard removes both the weight accounting and the ceiling.
    pub fn register_with_cache_limit(
        &self,
        weight_bytes: u64,
        cache_limit: Option<usize>,
    ) -> CacheLimitGuard {
        let id = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let id = state.next_id;
            state.next_id = state.next_id.saturating_add(1);
            state.profiles.insert(
                id,
                ModelCacheProfile {
                    weight_bytes,
                    cache_limit,
                },
            );
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

    /// Sum of every live private paged-KV pool registered with this
    /// coordinator. Used by load-time pool sizers so a second model does
    /// not treat another model's Metal buffers as free headroom. Includes a
    /// fixed restore-staging allowance per pool; MLX cannot see these retained
    /// shared Metal allocations either. Pool registration/update arguments
    /// remain KV bytes only, so growth cannot accidentally drop the allowance.
    pub fn registered_pool_bytes(&self) -> u64 {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state
            .pools
            .values()
            .copied()
            .fold(0u64, u64::saturating_add)
    }

    /// Register `pool_bytes` only if the live pool sum is still `expected_total`.
    /// `None` means another load reserved in between; the caller should resize.
    pub fn try_register_pool_if_total_eq(
        &self,
        expected_total: u64,
        pool_bytes: u64,
    ) -> Option<PoolCacheLimitGuard> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let current = state
            .pools
            .values()
            .copied()
            .fold(0u64, u64::saturating_add);
        if current != expected_total {
            return None;
        }
        let id = state.next_id;
        state.next_id = state.next_id.saturating_add(1);
        state.pools.insert(
            id,
            pool_bytes.saturating_add(mlx_paged_attn::RESTORE_STAGING_BYTES),
        );
        info!(
            "[cache_limit] register paged pool guard={} bytes={:.2} GB (live_pools={})",
            id,
            pool_bytes as f64 / ONE_GIB,
            state.pools.len(),
        );
        recompute_locked(&mut state);
        Some(PoolCacheLimitGuard { id })
    }

    /// Register a private paged-KV pool so the MLX freelist ceiling is
    /// debited by memory that MLX's own allocator counters cannot see. Reserve
    /// restore staging up front, avoiding callbacks under the pool upload locks.
    pub fn register_pool(&self, pool_bytes: u64) -> PoolCacheLimitGuard {
        let id = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let id = state.next_id;
            state.next_id = state.next_id.saturating_add(1);
            state.pools.insert(
                id,
                pool_bytes.saturating_add(mlx_paged_attn::RESTORE_STAGING_BYTES),
            );
            info!(
                "[cache_limit] register paged pool guard={} bytes={:.2} GB (live_pools={})",
                id,
                pool_bytes as f64 / ONE_GIB,
                state.pools.len(),
            );
            recompute_locked(&mut state);
            id
        };
        PoolCacheLimitGuard { id }
    }

    /// Replace a live paged pool's byte accounting in place. Dynamic
    /// (grow-on-demand) pools grow their Metal footprint after load, and the
    /// cap must debit the NEW total — the entry's lifetime is still owned by
    /// the guard from `register_pool`. No-op when the id is unknown (guard
    /// already dropped) or the byte total is unchanged.
    pub fn update_pool(&self, id: u64, new_bytes: u64) {
        let new_bytes = new_bytes.saturating_add(mlx_paged_attn::RESTORE_STAGING_BYTES);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(bytes) = state.pools.get_mut(&id) else {
            return;
        };
        if *bytes == new_bytes {
            return;
        }
        *bytes = new_bytes;
        info!(
            "[cache_limit] update paged pool guard={} bytes={:.2} GB (live_pools={})",
            id,
            new_bytes as f64 / ONE_GIB,
            state.pools.len(),
        );
        recompute_locked(&mut state);
    }

    fn unregister(&self, id: u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.profiles.remove(&id).is_some() {
            // Recompute the surviving policy, restoring the pre-ceiling cap
            // when the last temporary model constraint disappears.
            recompute_locked(&mut state);
        }
    }

    fn unregister_pool(&self, id: u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.pools.remove(&id).is_some() {
            recompute_locked(&mut state);
        }
    }
}

/// RAII token returned from [`CacheLimitCoordinator::register`]. Dropping
/// it unregisters the delta and triggers a recompute so the cap shrinks
/// back down when a model unloads.
///
/// Each generative model keeps one of these with the owner of its native
/// weights. That is normally the JavaScript model wrapper; models that expose
/// independently retained stream handles keep it in their model-thread state
/// instead. Dropping the final owner of the weights drops the guard and
/// unregisters the model.
pub struct CacheLimitGuard {
    id: u64,
}

pub struct PoolCacheLimitGuard {
    id: u64,
}

impl PoolCacheLimitGuard {
    /// Coordinator id this pool entry is registered under. Captured by the
    /// growth notifier closure (id is `Copy`, no guard sharing needed) so a
    /// grown pool can `update_pool` its own entry while the guard stays put.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for PoolCacheLimitGuard {
    fn drop(&mut self) {
        coordinator().unregister_pool(self.id);
    }
}

impl Drop for CacheLimitGuard {
    fn drop(&mut self) {
        coordinator().unregister(self.id);
    }
}

/// RAII token from [`CacheLimitCoordinator::push_decode_limit`]. Dropping
/// it removes the turn's ceiling and recomputes the effective cap, so the
/// load-time ceiling is back in force the moment the decode loop ends.
#[must_use = "dropping the guard immediately lifts the decode-time ceiling"]
pub struct DecodeCacheLimitGuard {
    id: u64,
}

impl Drop for DecodeCacheLimitGuard {
    fn drop(&mut self) {
        coordinator().pop_decode_limit(self.id);
    }
}

/// Smallest decode-time free-pool ceiling (bytes) that still lets a
/// speculative decode loop reuse every buffer it frees.
///
/// `transient_bytes` is the caller's estimate of the short-lived bytes one
/// verify/propose cycle allocates and frees (logits, per-layer tapes, one
/// layer's scratch, two-pass attention partials — see
/// `qwen3_5::dflash2_decode::dflash2_cycle_transient`).
///
/// ```text
/// cap = max(2 × transient, DECODE_CACHE_LIMIT_FLOOR)
/// ```
///
///   * `2×` — `BufferCache::reuse_from_cache` only hands back a buffer
///     whose size is in `[size, min(2·size, size + 2 pages))`, and the
///     sizes jitter with the accepted-draft count, so the pool must hold
///     the previous cycle's buffers while the current cycle's slightly
///     different sizes also land in it. One cycle's worth would evict
///     buffers the very next cycle wants. For contexts long enough to have
///     attention partials this puts the cap at or above the unbounded
///     pool's own end-of-turn size (Qwen3.8-27B: 525 MiB cap vs 340 MiB
///     measured at 6 K, 2.0 GiB vs 360 MiB at 32 K), i.e. the ceiling then
///     binds nothing — by design, since a 128 MiB cap at 6 K measured
///     +1.2% mean / +0.6% paired median and a 256 MiB cap could not be
///     resolved within ±1% in the time available, while the memory it
///     would return there is only tens of MiB.
///   * floor [`DECODE_CACHE_LIMIT_FLOOR`] (128 MiB) — the cap measured for
///     Qwen3.8-27B DFlash2 at a 1 K prompt (transient estimate ≈ 46 MiB,
///     no partials): in-process ABAB, 4 pairs, ms/cycle −0.8% mean / −3.7%
///     paired median vs the unbounded pool, identical output hashes, equal
///     Metal command-buffer counts (no allocation storm), end-of-turn pool
///     1.2 GiB → ≈ 190 MiB. That 1.2 GiB is dead weight: the draft's
///     sliding-window context grows every cycle until it is full, and
///     `reuse_from_cache` never matches a grown request to the smaller
///     buffer it just freed, so those pile up until the 256-token
///     `clear_cache()`. Small geometries whose estimate falls under the
///     floor are still safe: it is tiny next to any load-time cap.
///
/// Pure function; see the unit tests for the contract.
pub fn decode_cache_limit(transient_bytes: u64) -> u64 {
    transient_bytes
        .saturating_mul(2)
        .max(DECODE_CACHE_LIMIT_FLOOR)
}

/// Floor for [`decode_cache_limit`], see its docs.
pub const DECODE_CACHE_LIMIT_FLOOR: u64 = 128 << 20;

/// Access the process-wide coordinator, initializing it on first use.
pub fn coordinator() -> &'static CacheLimitCoordinator {
    static INSTANCE: OnceLock<CacheLimitCoordinator> = OnceLock::new();
    INSTANCE.get_or_init(CacheLimitCoordinator::new)
}

/// Process-wide lock serializing paged-KV pool growth AND load-time pool
/// reservation across models.
///
/// Growth side: each loaded model runs `try_grow_pool` on its own
/// `"mlx-model"` thread, and the headroom probe's sibling accounting
/// (`registered_pool_bytes() − own_live`) is only sound when no other grow
/// is in flight: two concurrent probes would both read the pre-grow totals,
/// both pass, and both `grow_to`, transiently holding old+new buffers whose
/// combined peak can exceed the unified-memory budget.
///
/// Load side: a loader probes live Metal headroom
/// (`load_time_pool_sizing`), allocates its `LayerKVPool`, and only then
/// debits the total via `register_pool`. Without this lock a concurrent
/// grow (or another load) reads sibling totals that miss the in-flight
/// reservation — and the loader reads totals that miss the in-flight grow
/// — so the combined old+new+new-load transient can exceed the budget.
/// Dynamic-pool loaders (qwen3_5, qwen3_5_moe) therefore hold this lock
/// across the whole sizing-probe → `LayerKVPool::new` allocation →
/// `register_pool` span, making the sibling totals a fresh read on both
/// sides.
///
/// Lock ordering: this is the OUTERMOST lock in the growth and
/// load-reservation paths — it is acquired BEFORE the coordinator mutex
/// (`register_pool` / `update_pool` / `registered_pool_bytes`), the
/// `BlockAllocator` mutex, and the pool's internal locks, so the order is
/// always growth lock → coordinator mutex and growth lock → allocator
/// mutex → pool locks. No code path may hold the coordinator mutex, the
/// allocator mutex, or a pool lock and then take this lock, and the lock
/// is never nested inside any of them.
pub(crate) fn pool_growth_lock() -> &'static Mutex<()> {
    static POOL_GROWTH_LOCK: Mutex<()> = Mutex::new(());
    &POOL_GROWTH_LOCK
}

struct CachePolicy {
    limit: Option<usize>,
    source: String,
    // Preserve the process override: explicit pins (including disabled auto)
    // bypass automatic turn limits, but model-specific ceilings still apply.
    allow_decode_limit: bool,
}

fn recompute_locked(state: &mut CoordState) {
    let policy = cache_policy(state);
    let ceiling = state.scoped_cache_ceiling(policy.allow_decode_limit);
    let source = match ceiling {
        Some(bytes) => format!(
            "{}; scoped ceiling={:.3} GiB",
            policy.source,
            bytes as f64 / ONE_GIB
        ),
        None => policy.source,
    };
    state
        .limit
        .update(policy.limit, ceiling, |bytes| apply_limit(bytes, &source));
}

/// Select the process policy independently of model and decode ceilings.
/// None means leave the allocator unmanaged, not a zero-byte pool.
fn cache_policy(state: &CoordState) -> CachePolicy {
    if let Ok(raw) = std::env::var(CACHE_LIMIT_ENV) {
        let trimmed = raw.trim();
        match trimmed.parse::<f64>() {
            Ok(gib) if gib <= 0.0 => {
                return CachePolicy {
                    limit: None,
                    source: format!("env {CACHE_LIMIT_ENV}={trimmed}; auto disabled"),
                    allow_decode_limit: false,
                };
            }
            Ok(gib) => {
                return CachePolicy {
                    limit: Some((gib * ONE_GIB).round() as usize),
                    source: format!("env {CACHE_LIMIT_ENV}={trimmed}"),
                    allow_decode_limit: false,
                };
            }
            Err(_) => {
                info!(
                    "[cache_limit] Ignoring unparseable {}={:?}, using auto formula",
                    CACHE_LIMIT_ENV, raw
                );
            }
        }
    }

    let summed_weights = state
        .profiles
        .values()
        .fold(0u64, |sum, p| sum.saturating_add(p.weight_bytes));
    let summed_pools = state
        .pools
        .values()
        .copied()
        .fold(0u64, u64::saturating_add);
    if summed_weights == 0 && summed_pools == 0 {
        return CachePolicy {
            limit: None,
            source: "no live model budget; restore prior independent cap if needed".into(),
            allow_decode_limit: true,
        };
    }
    let wired = WiredLimitContext::get_max_working_set_size() as u64;
    let limit = compute_cache_limit(summed_weights, summed_pools, wired);
    let source = if wired == 0 {
        format!(
            "auto (weights={:.1}GB, pools={:.1}GB, wired=0 → fallback cap={:.1}GB, live_guards={})",
            summed_weights as f64 / ONE_GIB,
            summed_pools as f64 / ONE_GIB,
            limit as f64 / ONE_GIB,
            state.profiles.len(),
        )
    } else {
        format!(
            "auto (weights={:.1}GB, pools={:.1}GB, overhead={:.1}GB, headroom={:.1}GB, wired={:.1}GB → cap={:.1}GB, live_guards={})",
            summed_weights as f64 / ONE_GIB,
            summed_pools as f64 / ONE_GIB,
            estimate_metal_overhead(wired) as f64 / ONE_GIB,
            estimate_user_headroom(wired) as f64 / ONE_GIB,
            wired as f64 / ONE_GIB,
            limit as f64 / ONE_GIB,
            state.profiles.len(),
        )
    };
    CachePolicy {
        limit: Some(limit as usize),
        source,
        allow_decode_limit: true,
    }
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
///     `max(weights * 3/2 - pool_bytes, 1 GiB)`.
///   - `weights + pool_bytes + overhead + headroom >= wired` → return
///     [`MIN_FREELIST_BYTES`] (tight-fit floor).
///   - otherwise → `wired - weights - pool_bytes - overhead - headroom`, clamped to
///     at least [`MIN_FREELIST_BYTES`].
fn compute_cache_limit(weights: u64, pool_bytes: u64, wired: u64) -> u64 {
    if wired == 0 {
        return weights
            .saturating_mul(3)
            .saturating_div(2)
            .saturating_sub(pool_bytes)
            .max(MIN_FREELIST_BYTES);
    }
    let overhead = estimate_metal_overhead(wired);
    let headroom = estimate_user_headroom(wired);
    let reserved = weights
        .saturating_add(pool_bytes)
        .saturating_add(overhead)
        .saturating_add(headroom);
    if wired <= reserved {
        return MIN_FREELIST_BYTES;
    }
    (wired - reserved).max(MIN_FREELIST_BYTES)
}

/// Push a freshly computed cap through `set_cache_limit`. Returns the prior cap
/// when the FFI succeeded so the caller can update `last_applied`; `None`
/// indicates the FFI caught a C++ exception (degraded Metal) and the cap
/// was NOT applied — the caller MUST leave `last_applied` untouched so
/// the next register/unregister cycle retries.
///
/// Logging:
///   - success → `info!` with the new cap, source, and previous cap.
///   - failure → `warn!` so an operator can grep logs for the explicit
///     failure reason instead of having to reason about a silent retry
///     loop.
#[must_use]
fn apply_limit(bytes: usize, source: &str) -> Option<usize> {
    match set_cache_limit(bytes as f64) {
        Ok(prev) => {
            info!(
                "[cache_limit] cache pool cap set to {:.2} GB ({}); previous = {:.2} GB",
                bytes as f64 / ONE_GIB,
                source,
                prev / ONE_GIB,
            );
            Some(prev as usize)
        }
        Err(err) => {
            warn!(
                "[cache_limit] set_cache_limit({:.2} GB, {}) FAILED ({}); cap NOT applied, will \
                 retry on next register/unregister",
                bytes as f64 / ONE_GIB,
                source,
                err,
            );
            None
        }
    }
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

    /// Serializes tests that mutate the process-wide pool registry.
    static POOL_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn model_and_decode_ceilings_restore_baseline_in_either_drop_order() {
        for (model, decode) in [(4, 2), (2, 4)] {
            for model_first in [true, false] {
                let coord = CacheLimitCoordinator::new();
                let mut state = coord.state.lock().unwrap();
                let mut actual = 8;
                state.profiles.insert(
                    1,
                    ModelCacheProfile {
                        weight_bytes: 0,
                        cache_limit: Some(model),
                    },
                );
                let ceiling = state.scoped_cache_ceiling(true);
                state
                    .limit
                    .update(None, ceiling, |n| Some(std::mem::replace(&mut actual, n)));
                state.decode_limits.insert(2, decode as u64);
                let ceiling = state.scoped_cache_ceiling(true);
                state
                    .limit
                    .update(None, ceiling, |n| Some(std::mem::replace(&mut actual, n)));
                assert_eq!(actual, model.min(decode));
                if model_first {
                    state.profiles.remove(&1);
                } else {
                    state.decode_limits.remove(&2);
                }
                let ceiling = state.scoped_cache_ceiling(true);
                state
                    .limit
                    .update(None, ceiling, |n| Some(std::mem::replace(&mut actual, n)));
                assert_eq!(actual, if model_first { decode } else { model });
                state.profiles.clear();
                state.decode_limits.clear();
                state
                    .limit
                    .update(None, None, |n| Some(std::mem::replace(&mut actual, n)));
                assert_eq!(actual, 8);
            }
        }
    }

    #[test]
    fn global_pin_bypasses_decode_caps_but_keeps_explicit_model_ceilings() {
        let _lock = ENV_LOCK.lock().unwrap();
        let coord = CacheLimitCoordinator::new();
        let mut state = coord.state.lock().unwrap();
        state.profiles.insert(
            1,
            ModelCacheProfile {
                weight_bytes: 0,
                cache_limit: Some((GB / 2) as usize),
            },
        );
        state.decode_limits.insert(2, GB / 4);
        for (value, expected_policy, allow_decode) in [
            ("2", Some((2 * GB) as usize), false),
            ("0", None, false),
            ("invalid", None, true),
        ] {
            let _env = EnvGuard::set(CACHE_LIMIT_ENV, value);
            let policy = cache_policy(&state);
            assert_eq!(policy.limit, expected_policy);
            assert_eq!(policy.allow_decode_limit, allow_decode);
            assert_eq!(
                state.scoped_cache_ceiling(policy.allow_decode_limit),
                Some((if allow_decode { GB / 4 } else { GB / 2 }) as usize),
            );
        }
    }

    #[test]
    fn model_ceilings_compose_independently_of_registration_order() {
        let coord = CacheLimitCoordinator::new();
        let mut state = coord.state.lock().unwrap();
        state.profiles.insert(
            1,
            ModelCacheProfile {
                weight_bytes: GB,
                cache_limit: Some(2),
            },
        );
        state.profiles.insert(
            2,
            ModelCacheProfile {
                weight_bytes: GB,
                cache_limit: None,
            },
        );
        state.profiles.insert(
            3,
            ModelCacheProfile {
                weight_bytes: GB,
                cache_limit: Some(1),
            },
        );
        assert_eq!(state.cache_ceiling(), Some(1));
        state.profiles.remove(&3);
        assert_eq!(state.cache_ceiling(), Some(2));
        state.profiles.remove(&1);
        assert_eq!(state.cache_ceiling(), None);
    }

    #[test]
    fn model_ceiling_restores_unmanaged_baseline_and_retries_failed_restore() {
        let mut state = AppliedCacheLimit::default();
        let mut actual = 8;
        state.update(None, Some(1), |n| Some(std::mem::replace(&mut actual, n)));
        assert_eq!(actual, 1);
        state.update(None, Some(2), |n| Some(std::mem::replace(&mut actual, n)));
        assert_eq!(actual, 2);
        state.update(None, None, |_| None);
        assert_eq!(state.last_applied, Some(2));
        assert_eq!(state.before_scoped_limits, Some(8));
        state.update(None, None, |n| Some(std::mem::replace(&mut actual, n)));
        assert_eq!(actual, 8);
        assert_eq!(state.before_scoped_limits, None);
    }

    #[test]
    fn model_ceiling_respects_tighter_policy_and_releases_to_surviving_policy() {
        let mut state = AppliedCacheLimit::default();
        let mut actual = 8;
        state.update(Some(2), Some(1), |n| {
            Some(std::mem::replace(&mut actual, n))
        });
        assert_eq!(actual, 1);
        state.update(Some(2), Some(4), |n| {
            Some(std::mem::replace(&mut actual, n))
        });
        assert_eq!(actual, 2);
        state.update(Some(3), None, |n| Some(std::mem::replace(&mut actual, n)));
        assert_eq!(actual, 3);
        assert_eq!(state.before_scoped_limits, None);
        state.update(None, None, |_| panic!("no temporary cap left to restore"));
    }

    #[test]
    fn failed_set_is_not_memoized_and_zero_is_a_real_generic_ceiling() {
        let mut state = AppliedCacheLimit::default();
        state.update(None, Some(0), |_| None);
        assert_eq!(state.last_applied, None);
        assert_eq!(state.before_scoped_limits, None);
        state.update(None, Some(0), |_| Some(8));
        assert_eq!(state.last_applied, Some(0));
        state.update(None, Some(0), |_| {
            panic!("unchanged cap should not be set again")
        });
        state.update(None, None, |n| {
            assert_eq!(n, 8);
            Some(0)
        });
        assert_eq!(state.last_applied, Some(8));
    }

    #[test]
    fn unchanged_target_still_tracks_ceiling_lifetime() {
        let mut state = AppliedCacheLimit::default();
        state.update(Some(2), None, |_| Some(8));
        state.update(Some(2), Some(2), |_| panic!("unchanged target"));
        assert_eq!(state.before_scoped_limits, Some(2));
        state.update(Some(2), None, |_| panic!("unchanged target"));
        assert_eq!(state.before_scoped_limits, None);
        state.update(None, None, |_| panic!("no temporary cap remains"));
    }

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
        let cap = compute_cache_limit(36 * GB, 0, 96 * GB);
        approx_eq_gb(cap, 45.6);
    }

    #[test]
    fn case2_48gb_wired_36gb_weights() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = max(4, 48/20=2.4) = 4 GB (floor)
        // headroom = max(4, 48/10=4.8) = 4.8 GB
        // cap = 48 - 36 - 4 - 4.8 = 3.2 GB
        let cap = compute_cache_limit(36 * GB, 0, 48 * GB);
        approx_eq_gb(cap, 3.2);
    }

    #[test]
    fn case3_48gb_wired_42gb_weights_hits_floor() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = 4 GB, headroom = 4.8 GB
        // reserved = 42 + 4 + 4.8 = 50.8 GB > 48 GB wired → floor
        let cap = compute_cache_limit(42 * GB, 0, 48 * GB);
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
        let cap = compute_cache_limit(36 * GB, 0, 192 * GB);
        approx_eq_gb(cap, 127.2);
    }

    #[test]
    fn case5_192gb_wired_10gb_weights() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        // overhead = 9.6 GB, headroom = 19.2 GB
        // cap = 192 - 10 - 9.6 - 19.2 = 153.2 GB
        let cap = compute_cache_limit(10 * GB, 0, 192 * GB);
        approx_eq_gb(cap, 153.2);
    }

    #[test]
    fn wired_zero_falls_back_to_weights_times_one_and_a_half() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        let cap = compute_cache_limit(10 * GB, 0, 0);
        // 10 * 3 / 2 = 15 GB
        approx_eq_gb(cap, 15.0);
    }

    #[test]
    fn paged_pool_bytes_debit_the_same_wired_budget() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_headroom_env();
        let without_pool = compute_cache_limit(36 * GB, 0, 96 * GB);
        let with_pool = compute_cache_limit(36 * GB, 8 * GB, 96 * GB);
        assert_eq!(without_pool.saturating_sub(with_pool), 8 * GB);
        approx_eq_gb(with_pool, 37.6);
    }

    #[test]
    fn registered_pool_bytes_tracks_live_guards() {
        let _lock = POOL_LOCK.lock().unwrap();
        let before = coordinator().registered_pool_bytes();
        let _g1 = coordinator().register_pool(3 * GB);
        let _g2 = coordinator().register_pool(5 * GB);
        assert_eq!(
            coordinator().registered_pool_bytes().saturating_sub(before),
            8 * GB + 2 * mlx_paged_attn::RESTORE_STAGING_BYTES
        );
    }

    #[test]
    fn try_register_pool_if_total_eq_rejects_stale_snapshot() {
        let _lock = POOL_LOCK.lock().unwrap();
        let before = coordinator().registered_pool_bytes();
        let _occupant = coordinator().register_pool(GB);
        assert!(
            coordinator()
                .try_register_pool_if_total_eq(before, GB)
                .is_none(),
            "stale sibling snapshot must not insert a second pool"
        );
        let current = coordinator().registered_pool_bytes();
        let accepted = coordinator().try_register_pool_if_total_eq(current, 2 * GB);
        assert!(accepted.is_some(), "matching snapshot must reserve");
        assert_eq!(
            coordinator()
                .registered_pool_bytes()
                .saturating_sub(current),
            2 * GB + mlx_paged_attn::RESTORE_STAGING_BYTES
        );
    }

    #[test]
    fn update_pool_replaces_a_live_entry_and_drops_unknown_ids() {
        let _lock = POOL_LOCK.lock().unwrap();
        let before = coordinator().registered_pool_bytes();
        let guard = coordinator().register_pool(GB);
        assert_eq!(
            coordinator().registered_pool_bytes().saturating_sub(before),
            GB + mlx_paged_attn::RESTORE_STAGING_BYTES,
            "register_pool must debit the initial pool"
        );
        coordinator().update_pool(guard.id(), 3 * GB);
        assert_eq!(
            coordinator().registered_pool_bytes().saturating_sub(before),
            3 * GB + mlx_paged_attn::RESTORE_STAGING_BYTES,
            "update_pool must replace the entry with the grown total"
        );
        // An id nobody owns (e.g. a guard that raced a model teardown) is
        // silently ignored — the coordinator never resurrects dropped pools.
        coordinator().update_pool(guard.id() + 1, GB);
        assert_eq!(
            coordinator().registered_pool_bytes().saturating_sub(before),
            3 * GB + mlx_paged_attn::RESTORE_STAGING_BYTES,
            "an unknown id must not add a new entry"
        );
        // Idempotent no-op: a repeated equal byte total must not change the
        // registry (and skips the recompute entirely).
        coordinator().update_pool(guard.id(), 3 * GB);
        assert_eq!(
            coordinator().registered_pool_bytes().saturating_sub(before),
            3 * GB + mlx_paged_attn::RESTORE_STAGING_BYTES,
            "an unchanged byte total must be a no-op"
        );
        drop(guard);
        assert_eq!(
            coordinator().registered_pool_bytes(),
            before,
            "dropping the guard must unregister the updated entry"
        );
    }

    // ── env override: MLX_GPU_HEADROOM_GB ─────────────────────────

    #[test]
    fn headroom_env_overrides_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "20");
        // overhead (96/20=4.8) = 4.8 GB, headroom forced to 20 GB
        // cap = 96 - 36 - 4.8 - 20 = 35.2 GB
        let cap = compute_cache_limit(36 * GB, 0, 96 * GB);
        approx_eq_gb(cap, 35.2);
    }

    #[test]
    fn headroom_env_zero_is_honoured() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "0");
        // overhead = 4.8, headroom forced to 0
        // cap = 96 - 36 - 4.8 - 0 = 55.2 GB
        let cap = compute_cache_limit(36 * GB, 0, 96 * GB);
        approx_eq_gb(cap, 55.2);
    }

    #[test]
    fn headroom_env_invalid_falls_back_to_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "not-a-number");
        // Invalid → default 10% applies → same as case 1.
        let cap = compute_cache_limit(36 * GB, 0, 96 * GB);
        approx_eq_gb(cap, 45.6);
    }

    #[test]
    fn headroom_env_negative_is_ignored() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(GPU_HEADROOM_ENV, "-5");
        // Negative → rejected → default 10% applies.
        let cap = compute_cache_limit(36 * GB, 0, 96 * GB);
        approx_eq_gb(cap, 45.6);
    }

    // ── decode-time ceiling policy ────────────────────────────────

    #[test]
    fn decode_cache_limit_is_twice_the_transient_above_the_floor() {
        // 100 MiB transient → 200 MiB (previous cycle's buffers stay
        // resident while the current cycle allocates).
        assert_eq!(decode_cache_limit(100u64 << 20), 200u64 << 20);
        assert_eq!(decode_cache_limit(300u64 << 20), 600u64 << 20);
    }

    #[test]
    fn decode_cache_limit_floors_small_transients() {
        // 46 MiB (Qwen3.8-27B short-prompt estimate) × 2 = 92 MiB < floor.
        assert_eq!(decode_cache_limit(46u64 << 20), DECODE_CACHE_LIMIT_FLOOR);
        assert_eq!(decode_cache_limit(0), DECODE_CACHE_LIMIT_FLOOR);
        // Exactly at the knee: 64 MiB × 2 == floor; one byte more lifts it.
        assert_eq!(decode_cache_limit(64u64 << 20), DECODE_CACHE_LIMIT_FLOOR);
        assert_eq!(
            decode_cache_limit((64u64 << 20) + 1),
            DECODE_CACHE_LIMIT_FLOOR + 2
        );
    }

    #[test]
    fn decode_cache_limit_never_overflows() {
        assert_eq!(decode_cache_limit(u64::MAX), u64::MAX);
        assert_eq!(decode_cache_limit(u64::MAX / 2 + 1), u64::MAX);
    }

    #[test]
    fn decode_limit_guards_compose_by_min_and_lift_on_drop() {
        let _lock = POOL_LOCK.lock().unwrap();
        let coord = coordinator();
        assert_eq!(coord.active_decode_limit(), None);
        let a = coord.push_decode_limit(256 << 20);
        assert_eq!(coord.active_decode_limit(), Some(256 << 20));
        let b = coord.push_decode_limit(128 << 20);
        assert_eq!(coord.active_decode_limit(), Some(128 << 20));
        drop(b);
        assert_eq!(coord.active_decode_limit(), Some(256 << 20));
        drop(a);
        assert_eq!(coord.active_decode_limit(), None);
    }

    #[test]
    fn decode_limit_of_zero_is_ignored() {
        let _lock = POOL_LOCK.lock().unwrap();
        let coord = coordinator();
        let guard = coord.push_decode_limit(0);
        assert_eq!(coord.active_decode_limit(), None);
        drop(guard);
        assert_eq!(coord.active_decode_limit(), None);
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
