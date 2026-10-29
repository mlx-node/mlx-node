//! Admission of the expert cache: how many cache
//! slots fit next to the fixed working set, from one host reading passed in
//! as a value, and the per-turn re-check that drops the whole cache bank to
//! staging-only when other processes grew (purgeable state is per whole
//! buffer, so a bank never shrinks in place).
//!
//! ```text
//! others = host_used - own                     own = MLX active + MLX pool - volatile banks
//! live   = physical - others - reserve
//! fixed  = dense + staging x layers + 1.2 (KV(ctx) + compute(ctx)) + io(ctx) + driver + pool
//! cache  = floor_to_slots(min(live - fixed, ws - fixed, env))
//! layers = 2 when the cache does not need the second layer's bytes, or when
//!          the cache left after it is >= SECOND_STAGING_MIN_CACHE; else 1
//! ```

use napi::{Error, Result};

use crate::utils::host::HostMemory;

pub const CACHE_ENV: &str = "MLX_DSV41_EXPERT_CACHE_GB";
pub const RESERVE_ENV: &str = "MLX_DSV41_ADMIT_RESERVE_GB";

const GIB: u64 = 1 << 30;
/// Host memory left to other processes.
pub const RESERVE: u64 = 12 * GIB;
/// MLX free-buffer pool allowance (the pool is capped at 1 GiB).
pub const POOL: u64 = 2 * GIB;
const DRIVER_MIN: u64 = 4 * GIB;
const IO_BASE: u64 = GIB / 4;
const IO_PER_ROW: u64 = 40 << 10;
/// Chord through T = 1K / 8K of the measured MLX prefill scratch (f32
/// activations); bounds the 16K point from above.
/// Last-row logits only (`Model::forward_last`): a caller that keeps
/// `[T, vocab]` f32 logits (`generate.rs`, test driver) needs T x 0.5 MB more.
const COMPUTE_BASE: u64 = 1_134_024_803;
const COMPUTE_PER_ROW: u64 = 772_038;
const KV_BASE: u64 = 10_485_760;
const KV_PER_ROW: u64 = 6_400;
/// A kept cache may overshoot the live budget by this much before the turn
/// drops it (the guard's host line sits 4 GiB below the reserve).
pub const KEEP_SLACK: u64 = 2 * GIB;
/// A kept cache is rebuilt larger only when the admission grew by this much.
pub const GROW_MIN: u64 = 8 * GIB;
/// The second staging layer (prefill read-ahead) takes its bytes from the
/// cache only while this much cache is left; below it decode slows by more
/// than 5 % (internal-SSD read rate; a slower drive raises the bar).
pub const SECOND_STAGING_MIN_CACHE: u64 = 17 * GIB;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("moe_stream budget: {msg}"))
}

fn gib(b: u64) -> f64 {
    b as f64 / GIB as f64
}

/// What the loaded model needs, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelSize {
    /// Resident dense arrays.
    pub dense: u64,
    /// One expert of every projection (one bank slot).
    pub slot: u64,
    /// Staging slots per layer.
    pub n_experts: u64,
    /// Context limit in tokens.
    pub ctx: u64,
}

/// The env knobs: the cache cap can only lower the cache, the reserve can
/// only grow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub reserve: u64,
    pub cache_cap: Option<u64>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            reserve: RESERVE,
            cache_cap: None,
        }
    }
}

fn gib_value(name: &str, raw: &str) -> Result<u64> {
    match raw.trim().parse::<f64>() {
        Ok(v) if (0.0..=1e6).contains(&v) => Ok((v * GIB as f64) as u64),
        _ => Err(bad(format!("{name}={raw:?}: expected GiB >= 0"))),
    }
}

impl Limits {
    pub fn from_values(cache_gb: Option<&str>, reserve_gb: Option<&str>) -> Result<Self> {
        fn set(v: Option<&str>) -> Option<&str> {
            v.filter(|s| !s.trim().is_empty())
        }
        let cache_cap = set(cache_gb).map(|v| gib_value(CACHE_ENV, v)).transpose()?;
        let reserve = match set(reserve_gb) {
            Some(v) => gib_value(RESERVE_ENV, v)?.max(RESERVE),
            None => RESERVE,
        };
        Ok(Self { reserve, cache_cap })
    }

    pub fn from_env() -> Result<Self> {
        let get = |k| std::env::var(k).ok();
        Self::from_values(get(CACHE_ENV).as_deref(), get(RESERVE_ENV).as_deref())
    }
}

/// This process's non-volatile bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Own {
    pub active: u64,
    /// MLX free-buffer pool.
    pub pool: u64,
    /// Volatile / empty purgeable bank bytes (inside `active`, not in `host_used`).
    pub volatile: u64,
}

impl Own {
    pub fn bytes(&self) -> u64 {
        (self.active + self.pool).saturating_sub(self.volatile)
    }

    pub fn measure() -> Self {
        Self {
            active: crate::array::get_active_memory() as u64,
            pool: crate::array::get_cache_memory() as u64,
            volatile: crate::array::purgeable::reclaimable_purgeable_bytes(),
        }
    }
}

/// Metal `max_recommended_working_set_size` (0 without Metal).
pub fn working_set() -> u64 {
    crate::stream::WiredLimitContext::get_max_working_set_size() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Terms {
    pub dense: u64,
    pub staging_layer: u64,
    pub kv: u64,
    pub compute: u64,
    pub io: u64,
    pub driver: u64,
    pub pool: u64,
}

impl Terms {
    pub fn new(m: &ModelSize, ws: u64) -> Self {
        let x12 = |b: u64| b + b / 5;
        Self {
            dense: m.dense,
            staging_layer: m.n_experts * m.slot,
            kv: x12(KV_BASE + KV_PER_ROW * m.ctx),
            compute: x12(COMPUTE_BASE + COMPUTE_PER_ROW * m.ctx),
            io: IO_BASE + IO_PER_ROW * m.ctx,
            driver: DRIVER_MIN.max(ws / 20),
            pool: POOL,
        }
    }

    pub fn fixed(&self, staging_layers: u64) -> u64 {
        self.dense
            + self.staging_layer * staging_layers
            + self.kv
            + self.compute
            + self.io
            + self.driver
            + self.pool
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Live,
    WorkingSet,
    Env,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    pub physical: u64,
    pub host_used: u64,
    pub own: u64,
    pub others: u64,
    pub reserve: u64,
    pub live: u64,
    pub ws: u64,
    pub terms: Terms,
    pub staging_layers: u64,
    pub fixed: u64,
    pub slot: u64,
    pub cache_slots: u64,
    pub bound: Bound,
}

impl Admission {
    pub fn cache_bytes(&self) -> u64 {
        self.cache_slots * self.slot
    }

    /// `min(live, ws)`.
    pub fn budget(&self) -> u64 {
        self.live.min(self.ws)
    }

    /// The Metal wired-limit request for this admission.
    pub fn wired_limit(&self) -> u64 {
        self.fixed + self.cache_bytes()
    }

    fn staging_only(mut self) -> Self {
        self.staging_layers = 1;
        self.fixed = self.terms.fixed(1);
        self.cache_slots = 0;
        self
    }
}

impl std::fmt::Display for Admission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t = &self.terms;
        write!(
            f,
            "physical {:.2} GiB, host_used {:.2}, own {:.2}, others {:.2}, reserve {:.2}, \
             live {:.2}, ws {:.2}; fixed {:.2} = dense {:.2} + staging {} x {:.2} + kv {:.2} \
             + compute {:.2} + io {:.2} + driver {:.2} + pool {:.2}; cache {} slots = {:.2} GiB \
             (bound {:?}); wired {:.2} GiB",
            gib(self.physical),
            gib(self.host_used),
            gib(self.own),
            gib(self.others),
            gib(self.reserve),
            gib(self.live),
            gib(self.ws),
            gib(self.fixed),
            gib(t.dense),
            self.staging_layers,
            gib(t.staging_layer),
            gib(t.kv),
            gib(t.compute),
            gib(t.io),
            gib(t.driver),
            gib(t.pool),
            self.cache_slots,
            gib(self.cache_bytes()),
            self.bound,
            gib(self.wired_limit()),
        )
    }
}

/// Admission at load (and the first half of every re-admit). Refuses when
/// the fixed working set with one staging layer does not fit.
pub fn admit(
    m: &ModelSize,
    limits: &Limits,
    host: &HostMemory,
    own: &Own,
    ws: u64,
) -> Result<Admission> {
    if m.slot == 0 || m.n_experts == 0 {
        return Err(bad("empty expert slot"));
    }
    if ws == 0 {
        return Err(bad("no Metal working set"));
    }
    let host_used = host.non_reclaimable();
    let own = own.bytes();
    let others = host_used.saturating_sub(own);
    let live = host
        .physical
        .saturating_sub(others)
        .saturating_sub(limits.reserve);
    let budget = live.min(ws);
    let terms = Terms::new(m, ws);
    let one = terms.fixed(1);
    if one > budget {
        return Err(bad(format!(
            "need {:.2} GiB fixed (dense {:.2} + staging {:.2} + kv/compute/io {:.2} + driver/pool \
             {:.2}) vs budget {:.2} GiB = min(live {:.2}, ws {:.2}); others {:.2} GiB, reserve {:.2} GiB",
            gib(one),
            gib(terms.dense),
            gib(terms.staging_layer),
            gib(terms.kv + terms.compute + terms.io),
            gib(terms.driver + terms.pool),
            gib(budget),
            gib(live),
            gib(ws),
            gib(others),
            gib(limits.reserve),
        )));
    }
    let cap = limits.cache_cap.unwrap_or(u64::MAX);
    let cache1 = (budget - one).min(cap) / m.slot * m.slot;
    let two = terms.fixed(2);
    let layers =
        if two + cache1 <= budget || (two <= budget && budget - two >= SECOND_STAGING_MIN_CACHE) {
            2
        } else {
            1
        };
    let room = budget - terms.fixed(layers);
    let (cache, bound) = if cap < room {
        (cap, Bound::Env)
    } else if live <= ws {
        (room, Bound::Live)
    } else {
        (room, Bound::WorkingSet)
    };
    let cache_slots = cache / m.slot;
    Ok(Admission {
        physical: host.physical,
        host_used,
        own,
        others,
        reserve: limits.reserve,
        live,
        ws,
        terms,
        staging_layers: layers,
        fixed: terms.fixed(layers),
        slot: m.slot,
        cache_slots,
        bound,
    })
}

/// What a turn does with the bank. A drop or rebuild frees the old bank
/// (after the previous turn's eval) and calls `clear_cache()` before the new
/// one is allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    Keep,
    /// Others grew: run this turn on one staging layer, cache 0.
    StagingOnly(Admission),
    /// Room came back (or grew by [`GROW_MIN`]): a new bank of this size.
    Rebuild(Admission),
}

/// Per-turn re-check against the current bank. Errs only when the fixed
/// working set alone no longer fits.
pub fn readmit(
    current: &Admission,
    m: &ModelSize,
    limits: &Limits,
    host: &HostMemory,
    own: &Own,
    ws: u64,
) -> Result<Turn> {
    let now = admit(m, limits, host, own, ws)?;
    let held = current.fixed + current.cache_bytes();
    if current.cache_slots == 0 && current.staging_layers == 1 {
        return Ok(if now.cache_slots > 0 || now.staging_layers > 1 {
            Turn::Rebuild(now)
        } else {
            Turn::Keep
        });
    }
    if held > ws.min(now.live.saturating_add(KEEP_SLACK)) {
        return Ok(Turn::StagingOnly(now.staging_only()));
    }
    if now.cache_bytes() >= current.cache_bytes() + GROW_MIN {
        return Ok(Turn::Rebuild(now));
    }
    Ok(Turn::Keep)
}

#[cfg(test)]
mod tests;
