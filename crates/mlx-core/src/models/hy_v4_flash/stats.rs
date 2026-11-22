//! Stats line (`MLX_HY4_STATS=<file>`, one JSON object per call) from the
//! expert bank's and the Engram table's counters, and the phase split of a
//! forward (off unless `set_profiling(true)`).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde_json::{Value, json};

use super::bank::{KqBank, KqStats};
use super::rows::F8Table;
use super::store::LANES_MAX;
use crate::array::MxArray;
pub use crate::models::deepseek_v41::stats::{host_pressure, rusage};

pub const STATS_ENV: &str = "MLX_HY4_STATS";
pub const PROFILE_ENV: &str = "MLX_HY4_PROFILE";

/// Phases of a forward. Each phase evaluates its outputs before its clock
/// stops, so phases do not overlap (an eval does not change bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Engram rows (read + mix).
    Engram,
    /// HC mixes, reductions and norms.
    Hc,
    /// Attention projections.
    AttnProj,
    /// Indexer keys, scores and top-k.
    Indexer,
    /// Sparse attention core.
    Sparse,
    /// Router GEMM + ids to the host.
    Router,
    /// Expert slots: plan + reads (blocked time).
    Fill,
    /// Routed expert gathers and the latent projections.
    Routed,
    /// Shared expert and dense FFN.
    Shared,
    /// Final norm + head.
    Head,
    /// The MTP layer.
    Mtp,
}

pub const PHASES: [Phase; 11] = [
    Phase::Engram,
    Phase::Hc,
    Phase::AttnProj,
    Phase::Indexer,
    Phase::Sparse,
    Phase::Router,
    Phase::Fill,
    Phase::Routed,
    Phase::Shared,
    Phase::Head,
    Phase::Mtp,
];

static PHASE_NS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
static PROFILE_EMITTED_NS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
static PROFILE_EMIT_LOCK: Mutex<()> = Mutex::new(());
static PROFILE_INIT: OnceLock<()> = OnceLock::new();
static PROFILE: AtomicBool = AtomicBool::new(false);

fn parse_profile_env(val: Option<&str>) -> bool {
    val.is_some_and(|v| {
        let t = v.trim();
        t == "1" || t.eq_ignore_ascii_case("true")
    })
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Engram => "engram",
            Phase::Hc => "hc",
            Phase::AttnProj => "attn_proj",
            Phase::Indexer => "indexer",
            Phase::Sparse => "sparse_attn",
            Phase::Router => "router",
            Phase::Fill => "fill",
            Phase::Routed => "routed",
            Phase::Shared => "shared",
            Phase::Head => "head",
            Phase::Mtp => "mtp",
        }
    }
}

pub fn profiling() -> bool {
    PROFILE_INIT.get_or_init(|| {
        if parse_profile_env(std::env::var(PROFILE_ENV).ok().as_deref()) {
            PROFILE.store(true, Ordering::Relaxed);
        }
    });
    PROFILE.load(Ordering::Relaxed)
}

/// Turns the phase split on or off for this process.
pub fn set_profiling(on: bool) {
    let _ = PROFILE_INIT.set(());
    PROFILE.store(on, Ordering::Relaxed);
}

/// Returns per-phase delta milliseconds since the previous emission and advances
/// the emission cursor (leaving `phase_seconds()` monotonic). Returns `None`
/// when no phase recorded time since the last emission.
pub fn take_profile_delta_ms() -> Option<[f64; 11]> {
    let _g = PROFILE_EMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut delta_ms = [0.0f64; 11];
    let mut any = false;
    for i in 0..PHASES.len() {
        let cur = PHASE_NS[i].load(Ordering::Relaxed);
        let prev = PROFILE_EMITTED_NS[i].swap(cur, Ordering::Relaxed);
        let diff = cur.saturating_sub(prev);
        if diff > 0 {
            any = true;
            delta_ms[i] = diff as f64 * 1e-6;
        }
    }
    any.then_some(delta_ms)
}

pub fn format_profile_line(label: &str, ms: &[f64; 11]) -> String {
    let total: f64 = ms.iter().sum();
    let mut out = format!("[HY4_PROF] stage={label} total_ms={total:.2}");
    for (p, &v) in PHASES.iter().zip(ms) {
        use std::fmt::Write as _;
        let _ = write!(out, " {}={v:.2}", p.name());
    }
    out
}

/// When profiling is active and any phase accrued time since the last emission,
/// prints a compact summary line to stderr and advances the cursor.
pub fn emit_profile(label: &str) {
    if !profiling() {
        return;
    }
    if let Some(ms) = take_profile_delta_ms() {
        eprintln!("{}", format_profile_line(label, &ms));
    }
}

/// A phase clock when profiling (the GPU is drained first, so the phase
/// does not pay for lazy work queued before it).
pub fn prof_start() -> Option<Instant> {
    profiling().then(|| {
        crate::array::synchronize();
        Instant::now()
    })
}

/// Evaluates `outs` and adds the time since `t` to `phase` (no-op when `t`
/// is `None`).
pub fn prof_end(phase: Phase, t: Option<Instant>, outs: &[&MxArray]) -> napi::Result<()> {
    if let Some(t) = t {
        if !outs.is_empty() {
            MxArray::eval_arrays_with_context(outs, "hy_v4_flash profile")?;
        }
        crate::array::synchronize();
        PHASE_NS[phase as usize].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    Ok(())
}

/// Seconds per phase so far (monotonic, process-wide), `PHASES` order.
pub fn phase_seconds() -> [f64; 11] {
    std::array::from_fn(|i| PHASE_NS[i].load(Ordering::Relaxed) as f64 * 1e-9)
}

/// Bank and Engram counters so far (monotonic); `staging_layers` is the
/// bank's current value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Layer steps that read anything.
    pub fills: u64,
    /// Experts read into slots (misses; all projections of one counted once).
    pub fill_slots: u64,
    pub fill_bytes: u64,
    /// Wall time of the reads, submit to last done.
    pub fill_ns: u64,
    pub read_ns: u64,
    pub repack_ns: u64,
    /// Expert uses (distinct experts per layer step) and cache hits among them.
    pub uses: u64,
    pub hits: u64,
    pub promoted: u64,
    pub promote_ns: u64,
    pub wait_ns: u64,
    pub split_steps: u64,
    pub split_mixed: u64,
    pub prefetch_slots: u64,
    pub prefetch_unused: u64,
    pub engram_bytes: u64,
    pub engram_ns: u64,
    pub staging_layers: u64,
}

impl Counters {
    /// Both classes summed; `engram` = `F8Table::read_stats` (bytes, ns).
    pub fn of(bank: &[KqStats; 2], engram: (u64, u64), staging_layers: usize) -> Self {
        let mut c = Self {
            engram_bytes: engram.0,
            engram_ns: engram.1,
            staging_layers: staging_layers as u64,
            ..Self::default()
        };
        for s in bank {
            c.fills += s.fills as u64;
            c.fill_slots += s.slots as u64;
            c.fill_bytes += s.bytes;
            c.fill_ns += s.ns;
            c.read_ns += s.read_ns;
            c.repack_ns += s.repack_ns;
            c.uses += s.uses as u64;
            c.hits += s.hits as u64;
            c.promoted += s.promoted as u64;
            c.promote_ns += s.promote_ns;
            c.wait_ns += s.wait_ns;
            c.split_steps += s.split_steps as u64;
            c.split_mixed += s.split_mixed as u64;
            c.prefetch_slots += s.prefetch_slots as u64;
            c.prefetch_unused += s.prefetch_unused as u64;
        }
        c
    }

    pub fn now(bank: &KqBank, table: &F8Table) -> Self {
        Self::of(&bank.stats(), table.read_stats(), bank.staging_layers())
    }

    pub fn since(&self, before: &Self) -> Self {
        let d = |a: u64, b: u64| a.saturating_sub(b);
        Self {
            fills: d(self.fills, before.fills),
            fill_slots: d(self.fill_slots, before.fill_slots),
            fill_bytes: d(self.fill_bytes, before.fill_bytes),
            fill_ns: d(self.fill_ns, before.fill_ns),
            read_ns: d(self.read_ns, before.read_ns),
            repack_ns: d(self.repack_ns, before.repack_ns),
            uses: d(self.uses, before.uses),
            hits: d(self.hits, before.hits),
            promoted: d(self.promoted, before.promoted),
            promote_ns: d(self.promote_ns, before.promote_ns),
            wait_ns: d(self.wait_ns, before.wait_ns),
            split_steps: d(self.split_steps, before.split_steps),
            split_mixed: d(self.split_mixed, before.split_mixed),
            prefetch_slots: d(self.prefetch_slots, before.prefetch_slots),
            prefetch_unused: d(self.prefetch_unused, before.prefetch_unused),
            engram_bytes: d(self.engram_bytes, before.engram_bytes),
            engram_ns: d(self.engram_ns, before.engram_ns),
            staging_layers: self.staging_layers,
        }
    }
}

/// Wave rows the histogram tells apart; its last bucket also takes every
/// wider wave.
pub const WAVE_ROWS: usize = 12;

static EVALS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static DEFERRED: RefCell<Vec<MxArray>> = const { RefCell::new(Vec::new()) };
    static DEFER_SCOPES: Cell<usize> = const { Cell::new(0) };
    #[cfg(test)]
    static THREAD_EVALS: Cell<u64> = const { Cell::new(0) };
}

/// Evaluates `outs` and every array [`defer`] holds, in one call: one
/// counted host sync of a forward.
pub fn eval(outs: &[&MxArray], what: &str) -> napi::Result<()> {
    EVALS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    THREAD_EVALS.with(|n| n.set(n.get() + 1));
    let held = DEFERRED.with(|d| std::mem::take(&mut *d.borrow_mut()));
    if held.is_empty() {
        return MxArray::eval_arrays_with_context(outs, what);
    }
    let all: Vec<&MxArray> = outs.iter().copied().chain(held.iter()).collect();
    MxArray::eval_arrays_with_context(&all, what)
}

/// Inside a [`DeferScope`], `a` is evaluated by the next [`eval`] or
/// [`flush`] of this thread; outside one, now.
pub fn defer(a: &MxArray, what: &str) -> napi::Result<()> {
    if DEFER_SCOPES.with(Cell::get) == 0 {
        return eval(&[a], what);
    }
    DEFERRED.with(|d| d.borrow_mut().push(a.clone()));
    Ok(())
}

/// Inside a [`DeferScope`], `outs` are evaluated by the next [`eval`] or
/// [`flush`] of this thread; outside one, now.
pub fn defer_many(outs: &[&MxArray], what: &str) -> napi::Result<()> {
    if outs.is_empty() {
        return Ok(());
    }
    if DEFER_SCOPES.with(Cell::get) == 0 {
        return eval(outs, what);
    }
    DEFERRED.with(|d| {
        let mut held = d.borrow_mut();
        held.extend(outs.iter().map(|a| (*a).clone()));
    });
    Ok(())
}

/// Evaluates what [`defer`] holds (no sync when it holds nothing).
pub fn flush(what: &str) -> napi::Result<()> {
    if DEFERRED.with(|d| d.borrow().is_empty()) {
        return Ok(());
    }
    eval(&[], what)
}

/// Submits what [`defer`] holds via `async_eval` without blocking the host.
pub fn async_flush() {
    let held = DEFERRED.with(|d| std::mem::take(&mut *d.borrow_mut()));
    if held.is_empty() {
        return;
    }
    let refs: Vec<&MxArray> = held.iter().collect();
    MxArray::async_eval_arrays(&refs);
}

/// Arrays [`defer`] holds on this thread.
pub fn deferred() -> usize {
    DEFERRED.with(|d| d.borrow().len())
}

/// While alive, [`defer`] holds arrays; the last scope to drop forgets
/// whatever is still held (a failed step's state is put back anyway).
pub struct DeferScope(());

impl DeferScope {
    pub fn enter() -> Self {
        DEFER_SCOPES.with(|n| n.set(n.get() + 1));
        Self(())
    }
}

impl Drop for DeferScope {
    fn drop(&mut self) {
        let left = DEFER_SCOPES.with(|n| {
            let v = n.get().saturating_sub(1);
            n.set(v);
            v
        });
        if left == 0 {
            DEFERRED.with(|d| d.borrow_mut().clear());
        }
    }
}

/// Host syncs of every forward so far (monotonic, process-wide).
pub fn evals() -> u64 {
    EVALS.load(Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn thread_evals() -> u64 {
    THREAD_EVALS.with(Cell::get)
}

/// One decode wave, as [`WaveLog::record`] takes it.
#[derive(Clone, Debug)]
pub struct WaveSample<'a> {
    /// Each owner's sequence and rows, in wave order.
    pub owners: &'a [(u32, usize)],
    /// Routed experts one row picks over the streamed layers.
    pub picks_per_row: u64,
    /// The bank's and the table's counters over the wave.
    pub counters: Counters,
    pub evals: u64,
    pub secs: f64,
}

impl WaveSample<'_> {
    pub fn rows(&self) -> usize {
        self.owners.iter().map(|&(_, r)| r).sum()
    }

    /// The stats file's line of the wave.
    pub fn json(&self) -> Value {
        let c = &self.counters;
        let rows = self.rows() as u64;
        json!({
            "stage": "wave",
            "owners": self.owners.len(),
            "rows": rows,
            "union_experts": c.uses,
            "row_experts": rows * self.picks_per_row,
            "served_rate": served_rate(c.uses, c.fill_slots),
            "miss_bytes": c.fill_bytes,
            "engram_bytes": c.engram_bytes,
            "read_gbps": c.fill_bytes as f64 / 1e9 / (c.fill_ns as f64 * 1e-9).max(1e-9),
            "evals": self.evals,
            "wave_ms": self.secs * 1e3,
        })
    }
}

/// Share of `uses` served without a read (0 without a use).
pub fn served_rate(uses: u64, reads: u64) -> f64 {
    if uses == 0 {
        0.0
    } else {
        uses.saturating_sub(reads) as f64 / uses as f64
    }
}

/// Decode-wave counters since load.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Waves {
    pub decode: u64,
    /// Waves by rows: `[i]` holds the waves of `i + 1` rows.
    pub rows: [u64; WAVE_ROWS],
    /// Distinct experts per layer step, summed (the bank's uses).
    pub union_experts: u64,
    /// Rows times the routed experts each row picks.
    pub row_experts: u64,
    pub miss_bytes: u64,
    /// Wave rows kept: one emitted token each.
    pub tokens: u64,
    pub evals: u64,
    /// By owners in the wave (`[n - 1]`): expert uses, and the uses served
    /// without a read.
    pub lane_uses: [u64; LANES_MAX],
    pub lane_served: [u64; LANES_MAX],
    /// Prefill slices run while another scheduled turn was open.
    pub prefill_stall_ns: u64,
}

impl Waves {
    pub fn miss_bytes_per_token(&self) -> f64 {
        self.miss_bytes as f64 / self.tokens.max(1) as f64
    }

    pub fn evals_per_wave(&self) -> f64 {
        self.evals as f64 / self.decode.max(1) as f64
    }

    /// `[n - 1]`: the share of the uses of waves of `n` owners served
    /// without a read.
    pub fn hit_rate_by_lanes(&self) -> [f64; LANES_MAX] {
        std::array::from_fn(|i| {
            served_rate(
                self.lane_uses[i],
                self.lane_uses[i].saturating_sub(self.lane_served[i]),
            )
        })
    }
}

/// [`Waves`] and the rows each owner's last decode wave wrote that a cut
/// may still take back (a verify's rejected rows, the unemitted tail).
#[derive(Debug, Default)]
pub struct WaveLog {
    pub waves: Waves,
    open: HashMap<u32, usize>,
}

impl WaveLog {
    pub fn record(&mut self, s: &WaveSample) {
        let w = &mut self.waves;
        let c = &s.counters;
        let rows = s.rows();
        w.decode += 1;
        w.rows[rows.clamp(1, WAVE_ROWS) - 1] += 1;
        w.union_experts += c.uses;
        w.row_experts += rows as u64 * s.picks_per_row;
        w.miss_bytes += c.fill_bytes;
        w.tokens += rows as u64;
        w.evals += s.evals;
        let lane = s.owners.len().clamp(1, LANES_MAX) - 1;
        w.lane_uses[lane] += c.uses;
        w.lane_served[lane] += c.uses.saturating_sub(c.fill_slots);
        for &(seq, r) in s.owners {
            self.open.insert(seq, r);
        }
        emit_profile("decode");
    }

    /// `seq` lost its top `rows` rows: those of its last wave among them
    /// emitted no token.
    pub fn cut(&mut self, seq: u32, rows: usize) {
        let Some(open) = self.open.get_mut(&seq) else {
            return;
        };
        let back = rows.min(*open);
        *open -= back;
        self.waves.tokens = self.waves.tokens.saturating_sub(back as u64);
    }

    pub fn stalled(&mut self, ns: u64) {
        self.waves.prefill_stall_ns += ns;
    }

    pub fn forget(&mut self, seq: u32) {
        self.open.remove(&seq);
    }

    pub fn clear_open(&mut self) {
        self.open.clear();
    }
}

/// One stats line's numbers.
#[derive(Clone, Debug)]
pub struct Line<'a> {
    pub stage: &'a str,
    pub prompt_tok: usize,
    pub gen_tok: usize,
    pub ttft_s: f64,
    /// Seconds of each decode call.
    pub decode_s: &'a [f64],
    /// Since the turn start.
    pub counters: Counters,
    pub wall_s: f64,
    pub cache_slots: [usize; 2],
    pub ctx: usize,
    pub chunk: usize,
    pub peak_footprint: u64,
    /// MLX free-buffer pool (`crate::array::get_cache_memory`).
    pub mlx_cache_bytes: f64,
    pub swapout_delta: i64,
    pub memstatus_min: Option<u64>,
}

impl Line<'_> {
    pub fn json(&self) -> Value {
        const GIB: f64 = (1u64 << 30) as f64;
        let c = &self.counters;
        let mut d = self.decode_s.to_vec();
        d.sort_by(f64::total_cmp);
        let med = d.get(d.len() / 2).copied().unwrap_or(0.0);
        let (io_s, eng_s) = (c.fill_ns as f64 * 1e-9, c.engram_ns as f64 * 1e-9);
        let toks = self.gen_tok.max(1) as f64;
        let uses = c.uses.max(1) as f64;
        json!({
            "stage": self.stage,
            "prompt_tok": self.prompt_tok,
            "gen_tok": self.gen_tok,
            "ttft_s": self.ttft_s,
            "prefill_tok_s": self.prompt_tok as f64 / self.ttft_s.max(1e-9),
            "decode_tok_s_median": if med > 0.0 { 1.0 / med } else { 0.0 },
            "hit_rate_uses": c.hits as f64 / uses,
            // every use served without a read (cache hits and still-valid
            // staging copies); unused prefetches are excluded from uses
            "hit_rate_bytes": 1.0 - c.fill_slots.saturating_sub(c.prefetch_unused) as f64 / uses,
            "miss_gb_per_tok": c.fill_bytes as f64 / 1e9 / toks,
            "read_gbps": c.fill_bytes as f64 / 1e9 / io_s.max(1e-9),
            "t_c_ms": (self.wall_s - io_s - eng_s) * 1e3 / toks,
            "t_io_ms": io_s * 1e3 / toks,
            "t_io_wait_ms": c.wait_ns as f64 * 1e-6 / toks,
            "t_read_ms": c.read_ns as f64 * 1e-6 / toks,
            "t_repack_ms": c.repack_ns as f64 * 1e-6 / toks,
            "t_engram_ms": eng_s * 1e3 / toks,
            "split_steps": c.split_steps,
            "split_mixed": c.split_mixed,
            "prefetch_slots": c.prefetch_slots,
            "prefetch_unused": c.prefetch_unused,
            "promoted": c.promoted,
            "bytes_read_total": c.fill_bytes + c.engram_bytes,
            "engram_bytes": c.engram_bytes,
            "cache_slots": self.cache_slots[0] + self.cache_slots[1],
            "cache_slots_by_class": self.cache_slots,
            "staging_layers": c.staging_layers,
            "ctx": self.ctx,
            "prefill_chunk": self.chunk,
            "peak_footprint_gib": self.peak_footprint as f64 / GIB,
            "mlx_cache_gib": self.mlx_cache_bytes / GIB,
            "swapout_delta_mib": self.swapout_delta as f64 / (1u64 << 20) as f64,
            "memstatus_min": self.memstatus_min,
            "cold": c.hits == 0,
        })
    }
}

/// Appends `line` as one JSON line to `path` (no-op for `None`).
pub fn emit_to(path: Option<&OsStr>, line: &Value) -> std::io::Result<()> {
    let Some(p) = path else {
        return Ok(());
    };
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(p)?;
    writeln!(f, "{line}")
}

/// Appends `line` to `$MLX_HY4_STATS` when set.
pub fn emit(line: &Value) -> std::io::Result<()> {
    emit_to(std::env::var_os(STATS_ENV).as_deref(), line)
}

#[cfg(test)]
mod tests;
