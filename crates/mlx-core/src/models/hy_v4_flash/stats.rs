//! Stats line (`MLX_HY4_STATS=<file>`, one JSON object per call) from the
//! expert bank's and the Engram table's counters, and the phase split of a
//! forward (off unless `set_profiling(true)`).

use std::ffi::OsStr;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use serde_json::{Value, json};

use super::bank::{KqBank, KqStats};
use super::rows::F8Table;
use crate::array::MxArray;
pub use crate::models::deepseek_v41::stats::{host_pressure, rusage};

pub const STATS_ENV: &str = "MLX_HY4_STATS";

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
static PROFILE: AtomicBool = AtomicBool::new(false);

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
    PROFILE.load(Ordering::Relaxed)
}

/// Turns the phase split on or off for this process.
pub fn set_profiling(on: bool) {
    PROFILE.store(on, Ordering::Relaxed);
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
            engram_bytes: d(self.engram_bytes, before.engram_bytes),
            engram_ns: d(self.engram_ns, before.engram_ns),
            staging_layers: self.staging_layers,
        }
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
            // staging copies)
            "hit_rate_bytes": 1.0 - c.fill_slots as f64 / uses,
            "miss_gb_per_tok": c.fill_bytes as f64 / 1e9 / toks,
            "read_gbps": c.fill_bytes as f64 / 1e9 / io_s.max(1e-9),
            "t_c_ms": (self.wall_s - io_s - eng_s) * 1e3 / toks,
            "t_io_ms": io_s * 1e3 / toks,
            "t_read_ms": c.read_ns as f64 * 1e-6 / toks,
            "t_repack_ms": c.repack_ns as f64 * 1e-6 / toks,
            "t_engram_ms": eng_s * 1e3 / toks,
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
