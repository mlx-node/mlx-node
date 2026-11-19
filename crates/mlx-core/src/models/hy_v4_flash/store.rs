//! Store: the Hy4 trunk from its GGUF shards. Resident weights are read once
//! (`weights::Resident`), routed experts stream through the two-class
//! K-quant bank, Engram rows come from the F8 table on shard 2.
//!
//! The admission picks the context and the prefill chunk at load:
//!
//! ```text
//! ctx = max_context, max_context / 2, ... >= MIN_CONTEXT
//!   chunk = the largest of CHUNKS admitted with cache >= second_staging_min_cache,
//!           else the smallest admitted, else the next ctx
//! cache bytes -> class slots in proportion to the classes' layers
//! ```
//!
//! A staging layer is charged 1024 slots (512 per class) at the larger
//! slot: 2 layers = 2 x 1024 x 4,663,296 B = 8.89 GiB, above the 8.35 GiB
//! the bank allocates.
//!
//! Lanes (sequences decoded together) share one KV allowance sized for the
//! admitted context; each lane past the first is charged its floor and its
//! draft's pooled keys (`lane_bytes`). The context and chunk are picked for
//! one lane; the load then admits `min(LANES_MAX, MLX_SCHED_MAX_NUM_SEQS,
//! row_cap)` lanes at them (one when no wave fits), fewer when only fewer
//! fit, so lanes never shrink the window or the chunk grid. A decode wave holds at most `row_cap`
//! rows (`row_cap`), the rows its Q8 matrices take through `qmv_wide`.
//!
//! A turn is `begin_turn` .. `end_turn` (`Lease`): the start re-admits at the
//! same context and chunk and may drop the bank to staging-only or rebuild
//! it, then holds a wired-limit lease of fixed + cache bytes until the end,
//! when the bank becomes volatile.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use napi::{Error, Result};

use super::bank::{CLASSES, ClassLayout, KqBank, slot_bytes};
use super::cache::WINDOW_SLACK;
use super::config::{Attn, HyV4FlashConfig};
use super::model::Model;
use super::moe::ExpertSource;
use super::mtp::SEEDS;
use super::rows::{EngramRows, F8Table};
use super::scan::{Role, Scan};
use super::weights::{Form, Resident, form, resident_bytes, role_resident};
use crate::cache_limit::{CacheLimitGuard, coordinator};
use crate::models::deepseek_v41::store::POOL_LIMIT;
use crate::moe_stream::budget::{
    Admission, DSV41, Limits, ModelSize, Own, Profile, Turn, admit_with, readmit_with, working_set,
};
use crate::moe_stream::io::{DEFAULT_THREADS, IoPool, ShardSet};
use crate::stream::WiredLimitContext;
use crate::utils::host::{HostMemory, host_memory};

pub const CACHE_ENV: &str = "MLX_HY4_EXPERT_CACHE_GB";
pub const RESERVE_ENV: &str = "MLX_HY4_ADMIT_RESERVE_GB";
/// Lowers the admitted prefill chunk (a multiple of 256).
pub const CHUNK_ENV: &str = "MLX_HY4_PREFILL_CHUNK";
/// `0` leaves the MTP layer out of the admission.
pub const MTP_ENV: &str = "MLX_HY4_MTP";
/// Prefill chunk rows the admission may pick, largest first.
pub const CHUNKS: [usize; 3] = [8192, 4096, 2048];
/// The context ladder's floor.
pub const MIN_CONTEXT: usize = 16_384;
/// Most lanes a load admits.
pub const LANES_MAX: usize = 4;
/// Overrides MLX's qmv batch limit for every shape (`quantized.cpp`).
pub const SPLITK_ENV: &str = "MLX_QMM_SPLITK_MIN_M";
const CHUNK_ALIGN: usize = 256;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
/// RAM ladder entries (window clones and tails).
pub const LADDER_BYTES: u64 = 150 * MIB;
/// Gathered keys of the sparse attention core.
const SPARSE_BYTES: u64 = GIB;
/// Indexer score sub-chunks.
const INDEX_BYTES: u64 = 256 * MIB;
/// Per context token: index ids and scores of a prefill chunk.
const COMPUTE_PER_CTX: u64 = 128;
/// Disk-tier host staging and copy queue.
const IO_FIXED: u64 = 160 * MIB;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 store: {msg}"))
}

/// Hy4's admission constants at `kv_per_ctx` KV bytes per token. The
/// compute and io base and per-row terms are DS V4.1's measured values:
/// UNVERIFIED for Hy4 until its prefill peaks are measured.
pub fn profile(kv_per_ctx: u64) -> Profile {
    Profile {
        cache_env: CACHE_ENV,
        reserve_env: RESERVE_ENV,
        compute_base: DSV41.compute_base,
        compute_per_row: DSV41.compute_per_row,
        compute_fixed: SPARSE_BYTES + INDEX_BYTES,
        compute_per_ctx: COMPUTE_PER_CTX,
        kv_base: DSV41.kv_base + LADDER_BYTES,
        kv_per_ctx,
        io_base: DSV41.io_base,
        io_per_row: DSV41.io_per_row,
        io_fixed: IO_FIXED,
        second_staging_min_cache: DSV41.second_staging_min_cache,
    }
}

/// KV bytes per context token: the f16 latent row and the f16 pooled index
/// keys of every indexed layer (the MTP layer's with `mtp`), and the u32 id.
pub fn kv_per_token(cfg: &HyV4FlashConfig, mtp: bool) -> u64 {
    let layers = if mtp { &cfg.layers[..] } else { cfg.trunk() };
    let n = layers.iter().filter(|l| l.attn == Attn::Indexed).count() as u64;
    let row = cfg.mla.kv_row() as u64;
    let dim = cfg.indexer.head_dim as u64;
    let kpool = cfg.indexer.kpool.max(1) as u64;
    n * row * 2 + n * dim * 2 / kpool + 4
}

/// `profile` with `lanes - 1` lane floors added to the KV base.
pub fn lane_profile(p: &Profile, lanes: usize, floor: u64) -> Profile {
    Profile {
        kv_base: p.kv_base + lanes.saturating_sub(1) as u64 * floor,
        ..*p
    }
}

/// Bytes a lane holds beside the shared KV allowance: the f16 window ring
/// (`window + WINDOW_SLACK` rows) of every sliding layer, the f32 indexer
/// tail (`kpool - 1 + WINDOW_SLACK` rows) of every indexed layer (the MTP
/// layer's with `mtp`) and, with `mtp`, the `SEEDS` f32 seed rows of the
/// live draft.
pub fn lane_floor(cfg: &HyV4FlashConfig, mtp: bool) -> u64 {
    let layers = if mtp { &cfg.layers[..] } else { cfg.trunk() };
    let count = |a: Attn| layers.iter().filter(|l| l.attn == a).count() as u64;
    let windows =
        count(Attn::Sliding) * (cfg.window + WINDOW_SLACK) as u64 * cfg.mla.kv_row() as u64 * 2;
    let tail = (cfg.indexer.kpool.saturating_sub(1) + WINDOW_SLACK) as u64;
    let tails = count(Attn::Indexed) * tail * cfg.indexer.head_dim as u64 * 4;
    let seeds = if mtp {
        SEEDS as u64 * cfg.dim as u64 * 4
    } else {
        0
    };
    windows + tails + seeds
}

/// `lane_floor` and, with `mtp`, the pooled index keys of the draft's MTP
/// layer for `ctx` positions, which its first write allocates whole: what
/// one lane past the first adds beside the shared KV allowance.
pub fn lane_bytes(cfg: &HyV4FlashConfig, mtp: bool, ctx: usize) -> u64 {
    let pooled = if mtp {
        (ctx / cfg.indexer.kpool.max(1)) as u64 * cfg.indexer.head_dim as u64 * 2
    } else {
        0
    };
    lane_floor(cfg, mtp) + pooled
}

/// The most lanes up to `want` (at least 1) whose admission `admit(lanes)`
/// passes, with that admission; one lane's refusal otherwise.
pub fn plan_lanes<T>(want: usize, mut admit: impl FnMut(usize) -> Result<T>) -> Result<(usize, T)> {
    let mut lanes = want.max(1);
    loop {
        match admit(lanes) {
            Ok(t) => return Ok((lanes, t)),
            Err(e) if lanes == 1 => return Err(e),
            Err(e) => {
                tracing::info!(
                    "hy_v4_flash admission refused {lanes} lanes ({}); trying {}",
                    e.reason,
                    lanes - 1
                );
                lanes -= 1;
            }
        }
    }
}

/// The most lanes up to `want` admitted (`admit(lanes)`) at the context and
/// chunk of `one`, the one-lane pick.
pub fn lanes_at(
    one: Pick,
    want: usize,
    mut admit: impl FnMut(usize) -> Result<Admission>,
) -> (usize, Pick) {
    plan_lanes(want, |n| match n {
        1 => Ok(one),
        _ => admit(n).map(|admission| Pick { admission, ..one }),
    })
    .unwrap_or((1, one))
}

/// MLX's `get_qmv_batch_limit(D, O)` (`backend/metal/quantized.cpp`) on the
/// Metal architecture `arch` (`applegpu_g16s`: generation from the two
/// digits before the size letter, as `metal/device.cpp`): from this many
/// rows a transposed `quantized_matmul` of `k` inputs and `n` outputs
/// leaves the matvec kernels.
pub fn qmv_batch_limit(arch: &str, k: usize, n: usize) -> usize {
    let b = arch.as_bytes();
    let digit = |back: usize| {
        b.len()
            .checked_sub(back)
            .and_then(|i| b.get(i))
            .filter(|c| c.is_ascii_digit())
            .map_or(0, |c| usize::from(c - b'0'))
    };
    let generation = if b.len() >= 3 {
        digit(3) * 10 + digit(2)
    } else {
        0
    };
    let size = b.last().copied().unwrap_or(0);
    let class = if k <= 2048 && n <= 2048 {
        0
    } else if k <= 4096 && n <= 4096 {
        1
    } else {
        2
    };
    let table: [usize; 3] = if generation >= 17 && size != b'd' {
        [33, 25, 13]
    } else if generation >= 15 && size != b'd' {
        [13, 15, 13]
    } else if size == b'd' {
        [32, 18, 12]
    } else if generation >= 13 {
        [
            14,
            10,
            if generation == 13 && size == b's' {
                7
            } else {
                6
            },
        ]
    } else {
        [18, 12, 10]
    };
    table[class]
}

/// `MLX_QMM_SPLITK_MIN_M` read as MLX does (`std::atoi`, used when > 0).
pub fn splitk_min_m(raw: Option<&str>) -> Option<usize> {
    let s = raw?.trim_start();
    let (neg, rest) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let end = rest
        .bytes()
        .position(|c| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let v = rest[..end].parse::<i32>().ok()?;
    (!neg && v > 0).then_some(v as usize)
}

/// `(inputs, outputs)` of every matrix the model runs through
/// `quantized_matmul`: Q8_0 held as `Dense::Qmm` (a rank-3 tensor per head
/// group), and with `mtp` the MTP layer's BF16 matrices, held the same way.
pub fn q8_shapes(scan: &Scan, mtp: bool) -> Result<Vec<(usize, usize)>> {
    let mut shapes = Vec::new();
    for (role, held) in [(Role::Dense, Form::Qmm), (Role::Mtp, Form::Bf16)] {
        if role == Role::Mtp && !mtp {
            continue;
        }
        for (_, loc) in scan.with_role(role) {
            if let [k, n, ..] = loc.info.dims[..]
                && form(role, &loc.info)? == held
            {
                shapes.push((k as usize, n as usize));
            }
        }
    }
    shapes.sort_unstable();
    shapes.dedup();
    Ok(shapes)
}

/// Rows a decode wave may hold: one below the smallest qmv batch limit of
/// `shapes` on `arch` (`splitk` overrides every limit), so every Q8 call of
/// 2..=cap rows runs `qmv_wide` on the generations that have it. Below 2
/// no wave fits.
pub fn row_cap(arch: &str, splitk: Option<usize>, shapes: &[(usize, usize)]) -> Result<usize> {
    shapes
        .iter()
        .map(|&(k, n)| splitk.unwrap_or_else(|| qmv_batch_limit(arch, k, n)))
        .min()
        .map(|m| m.saturating_sub(1))
        .ok_or_else(|| bad("no Q8 matrix to size the wave by"))
}

/// The Metal architecture name MLX's kernels route by (it honours
/// `MLX_METAL_GPU_ARCH`).
fn routing_arch() -> Result<String> {
    let mut buf = [0 as std::ffi::c_char; 64];
    // SAFETY: `buf` is writable for its length; the bridge writes a
    // NUL-terminated name into it or returns false.
    let ok = unsafe { mlx_sys::mlx_test_metal_architecture(false, buf.as_mut_ptr(), buf.len()) };
    if !ok {
        return Err(bad("no Metal architecture name"));
    }
    let name: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8(name).map_err(|e| bad(format!("Metal architecture name: {e}")))
}

/// Resident bytes of the MTP layer as it is held: routed experts affine-4
/// g64 (bf16 scales and biases), other BF16 matrices affine-8 g32
/// (`Dense::Qmm`), the rest as `weights::resident_bytes`. 0 without one.
pub fn mtp_bytes(scan: &Scan) -> Result<u64> {
    let add = |acc: u64, b: u64| {
        acc.checked_add(b)
            .ok_or_else(|| bad("MTP bytes overflow u64"))
    };
    let mut n = 0u64;
    for (_, loc) in scan.with_role(Role::Mtp) {
        let b = match form(Role::Mtp, &loc.info)? {
            Form::Bf16 => {
                let v = loc.info.num_elements()?;
                v + v / 4
            }
            _ => resident_bytes(Role::Mtp, &loc.info)?,
        };
        n = add(n, b)?;
    }
    if let Some(l) = scan.mtp_experts() {
        for loc in [&l.gate, &l.up, &l.down] {
            let v = loc.info.num_elements()?;
            n = add(n, v / 2 + v / 16)?;
        }
    }
    Ok(n)
}

/// Layers and bank bytes of one expert per streamed class (`CLASSES` order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Classes {
    pub layers: [usize; 2],
    pub slot: [u64; 2],
}

impl Classes {
    pub fn of(scan: &Scan) -> Result<Self> {
        let classes: Vec<_> = scan
            .trunk_experts()
            .iter()
            .map(|l| l.as_ref().map(|l| l.class))
            .collect();
        let layout = ClassLayout::new(&classes)?;
        let mut slot = [0u64; 2];
        for (s, &c) in slot.iter_mut().zip(&CLASSES) {
            *s = slot_bytes(&scan.cfg.moe, c)?;
            if *s == 0 {
                return Err(bad(format!("{c:?}: empty expert slot")));
            }
        }
        Ok(Self {
            layers: [layout.layers[0].len(), layout.layers[1].len()],
            slot,
        })
    }

    /// Cache slots per class for `bytes`, in proportion to the layer counts
    /// at each class's own slot size; the rest goes to the second class.
    pub fn split(&self, bytes: u64) -> [usize; 2] {
        let [la, lb] = self.layers.map(|l| l as u128);
        let [sa, sb] = self.slot;
        let w = la * u128::from(sa) + lb * u128::from(sb);
        if w == 0 || sa == 0 || sb == 0 {
            return [0, 0];
        }
        let a = (u128::from(bytes) * la / w) as u64;
        let b = if lb == 0 { 0 } else { (bytes - a * sa) / sb };
        [a as usize, b as usize]
    }

    /// Staging slots of every class with layers.
    fn staging_experts(&self, n_experts: usize) -> u64 {
        self.layers.iter().filter(|&&l| l > 0).count() as u64 * n_experts as u64
    }
}

/// What the admission charges, from the headers alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    /// Resident trunk arrays (`Role::Dense`).
    pub dense_trunk: u64,
    /// The MTP layer as held (`mtp_bytes`).
    pub dense_mtp: u64,
    pub classes: Classes,
    pub n_experts: usize,
}

impl Sizes {
    pub fn of(scan: &Scan) -> Result<Self> {
        Ok(Self {
            dense_trunk: role_resident(scan, Role::Dense)?,
            dense_mtp: mtp_bytes(scan)?,
            classes: Classes::of(scan)?,
            n_experts: scan.cfg.moe.n_experts,
        })
    }

    /// One staging layer covers both classes (`n_experts` slots each) at the
    /// larger slot.
    pub fn model(&self, mtp: bool, ctx: usize) -> ModelSize {
        ModelSize {
            dense: self.dense_trunk + if mtp { self.dense_mtp } else { 0 },
            slot: self.classes.slot[0].max(self.classes.slot[1]),
            n_experts: self.classes.staging_experts(self.n_experts),
            ctx: ctx as u64,
        }
    }
}

/// The admitted context, prefill chunk and admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick {
    pub ctx: usize,
    /// At most `ctx`.
    pub chunk: usize,
    pub admission: Admission,
}

/// The context ladder and the chunk rule over `admit_with`; a chunk is
/// charged at most `ctx` rows. Errs with the last refusal at the floor.
pub fn pick(
    p: &Profile,
    base: &ModelSize,
    max_ctx: usize,
    limits: &Limits,
    host: &HostMemory,
    own: &Own,
    ws: u64,
) -> Result<Pick> {
    let smallest = CHUNKS[CHUNKS.len() - 1];
    let mut ctx = max_ctx;
    loop {
        let m = ModelSize {
            ctx: ctx as u64,
            ..*base
        };
        let mut fallback = None;
        let mut refusal = None;
        for chunk in CHUNKS {
            match admit_with(p, &m, chunk.min(ctx) as u64, limits, host, own, ws) {
                Ok(a) if a.cache_bytes() >= p.second_staging_min_cache => {
                    return Ok(Pick {
                        ctx,
                        chunk: chunk.min(ctx),
                        admission: a,
                    });
                }
                Ok(a) if chunk == smallest => fallback = Some(a),
                Ok(_) => {}
                Err(e) => refusal = Some(e),
            }
        }
        if let Some(admission) = fallback {
            return Ok(Pick {
                ctx,
                chunk: smallest.min(ctx),
                admission,
            });
        }
        let why = refusal.map_or_else(String::new, |e| e.reason);
        let next = ctx / 2;
        if next < MIN_CONTEXT {
            return Err(bad(format!("refused at context {ctx}: {why}")));
        }
        tracing::info!("hy_v4_flash admission refused at context {ctx} ({why}); trying {next}");
        ctx = next;
    }
}

/// `MLX_HY4_PREFILL_CHUNK`: unset or blank = no cap.
pub fn chunk_cap(raw: Option<&str>) -> Result<Option<usize>> {
    let Some(v) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    match v.parse::<usize>() {
        Ok(c) if c >= CHUNK_ALIGN && c % CHUNK_ALIGN == 0 => Ok(Some(c)),
        _ => Err(bad(format!(
            "{CHUNK_ENV}={v:?}: expected a multiple of {CHUNK_ALIGN}"
        ))),
    }
}

/// How many cache slots the bank gets.
#[derive(Clone, Debug, PartialEq)]
pub enum CachePlan {
    /// Staging only, no admission.
    #[cfg(test)]
    Off,
    /// Exactly these cache slots and staging layers, no admission.
    #[cfg(test)]
    Slots { cache: [usize; 2], staging: usize },
    /// Admission at load and at every turn start, limits from the env.
    Admit,
    /// `Admit` with these limits.
    #[cfg(test)]
    AdmitWith(Limits),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// The load option `maxContextTokens`.
    pub max_context: Option<usize>,
    /// Charge the MTP layer (when the file has one).
    pub mtp: bool,
    pub chunk_cap: Option<usize>,
    /// Lanes wanted; the admission may give fewer.
    pub lanes: usize,
    pub plan: CachePlan,
    /// The load option `maxMemoryBytes`: the admission's budget stays
    /// within it.
    pub max_memory: Option<u64>,
}

impl Options {
    /// `serial`: one sequence at a time (`MLX_SERVE_FORCE_SERIAL` at load),
    /// else `min(LANES_MAX, MLX_SCHED_MAX_NUM_SEQS)` lanes.
    pub fn from_env(max_context: Option<usize>, serial: bool) -> Result<Self> {
        let mtp = std::env::var(MTP_ENV).ok().is_none_or(|v| v.trim() != "0");
        let lanes = if serial {
            1
        } else {
            crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(LANES_MAX)
        };
        Ok(Self {
            max_context,
            mtp,
            chunk_cap: chunk_cap(std::env::var(CHUNK_ENV).ok().as_deref())?,
            lanes,
            plan: CachePlan::Admit,
            max_memory: None,
        })
    }
}

#[cfg(test)]
thread_local! {
    /// The host reading the admission sees on this thread instead of the
    /// real one.
    pub(super) static TEST_HOST: std::cell::Cell<Option<(HostMemory, u64)>> =
        const { std::cell::Cell::new(None) };
}

/// The working set an admission may fill: `ws`, at most `max_memory`.
pub fn capped_ws(ws: u64, max_memory: Option<u64>) -> u64 {
    max_memory.map_or(ws, |m| m.min(ws))
}

/// Host memory and the Metal working set for an admission, capped at
/// `max_memory`.
fn host_reading(max_memory: Option<u64>) -> Result<(HostMemory, u64)> {
    #[cfg(test)]
    if let Some((h, ws)) = TEST_HOST.with(std::cell::Cell::get) {
        return Ok((h, capped_ws(ws, max_memory)));
    }
    let h = host_memory().ok_or_else(|| bad("no host memory reading for admission"))?;
    Ok((h, capped_ws(working_set(), max_memory)))
}

/// This model's bytes inside `host_used`: resident arrays, the sessions'
/// buffers `kv` (made at their first write and kept, already charged in
/// `fixed`) and the bank's resident pages (volatile between turns).
/// Scratch and the shared pool stay in `others` (conservative).
pub fn own(bank: &KqBank, dense: u64, kv: u64) -> Result<Own> {
    Ok(own_of(dense, kv, bank.resident_bytes()?, bank.in_turn()))
}

/// [`own`] from its parts.
pub fn own_of(dense: u64, kv: u64, resident: u64, in_turn: bool) -> Own {
    Own {
        active: dense + kv + resident,
        pool: 0,
        volatile: if in_turn { 0 } else { resident },
    }
}

struct TurnState {
    admission: Option<Admission>,
    wired: Option<WiredLimitContext>,
    guard: CacheLimitGuard,
    /// Bank switches: (cache slots, seconds to free the old bank and allocate).
    switches: Vec<(usize, f64)>,
    /// The last switch failed: the bank may hold no slots, so the next turn
    /// rebuilds even when the admission is kept.
    lost: bool,
}

/// The per-turn re-admission and wired-limit lease of one bank.
pub struct Lease {
    profile: Profile,
    size: ModelSize,
    rows: u64,
    classes: Classes,
    /// `None`: no admission (tests).
    limits: Option<Limits>,
    max_memory: Option<u64>,
    turn: Mutex<TurnState>,
}

impl Lease {
    /// `admission` is the one `bank` was built for; `rows` the prefill rows
    /// it was charged at.
    pub fn new(
        profile: Profile,
        size: ModelSize,
        rows: u64,
        classes: Classes,
        limits: Option<Limits>,
        admission: Option<Admission>,
        bank: &KqBank,
    ) -> Self {
        Self {
            profile,
            size,
            rows,
            classes,
            limits,
            max_memory: None,
            turn: Mutex::new(TurnState {
                admission,
                wired: None,
                guard: coordinator().register_with_cache_limit(bank.nbytes(), Some(POOL_LIMIT)),
                switches: Vec::new(),
                lost: false,
            }),
        }
    }

    /// Re-admissions stay within `max_memory` too.
    pub fn with_max_memory(mut self, max_memory: Option<u64>) -> Self {
        self.max_memory = max_memory;
        self
    }

    fn turn(&self) -> MutexGuard<'_, TurnState> {
        self.turn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn admission(&self) -> Option<Admission> {
        self.turn().admission
    }

    /// Every bank switch so far: (cache slots of both classes, seconds).
    pub fn switches(&self) -> Vec<(usize, f64)> {
        self.turn().switches.clone()
    }

    pub fn in_turn(&self) -> bool {
        self.turn().wired.is_some()
    }

    fn switch(&self, t: &mut TurnState, bank: &KqBank, a: &Admission) -> Result<()> {
        let t0 = Instant::now();
        let slots = self.classes.split(a.cache_bytes());
        t.lost = true;
        bank.rebuild(slots, a.staging_layers as usize)?;
        t.lost = false;
        t.guard = coordinator().register_with_cache_limit(bank.nbytes(), Some(POOL_LIMIT));
        let s = t0.elapsed().as_secs_f64();
        let n = slots[0] + slots[1];
        tracing::info!("hy_v4_flash bank switch to {slots:?} cache slots in {s:.2} s");
        t.switches.push((n, s));
        Ok(())
    }

    /// Turn start: make the bank non-volatile, re-admit, switch the bank if
    /// the admission says so, take the wired-limit lease. `kv`: bytes of
    /// every live session (`Session::nbytes`). Errs (bank volatile again)
    /// when the fixed working set alone no longer fits.
    pub fn begin_turn(&self, bank: &KqBank, dense: u64, kv: u64) -> Result<()> {
        let mut t = self.turn();
        if let Some(limits) = &self.limits
            && t.wired.is_none()
        {
            let cur = t.admission.ok_or_else(|| bad("no admission"))?;
            // non-volatile first: its resident pages are then in `host_used`
            // and counted as own
            bank.begin_turn()?;
            let next = (|| {
                let (h, ws) = host_reading(self.max_memory)?;
                let o = own(bank, dense, kv)?;
                match readmit_with(
                    &self.profile,
                    &cur,
                    &self.size,
                    self.rows,
                    limits,
                    &h,
                    &o,
                    ws,
                )? {
                    Turn::Keep if !t.lost => Ok(cur),
                    Turn::Keep => {
                        self.switch(&mut t, bank, &cur)?;
                        Ok(cur)
                    }
                    Turn::StagingOnly(a) | Turn::Rebuild(a) => {
                        self.switch(&mut t, bank, &a)?;
                        tracing::info!("hy_v4_flash re-admission: {a}");
                        Ok(a)
                    }
                }
            })();
            let next = match next {
                Ok(n) => n,
                Err(e) => {
                    let _ = bank.end_turn();
                    return Err(e);
                }
            };
            t.admission = Some(next);
            t.wired = Some(WiredLimitContext::bounded(
                usize::try_from(next.wired_limit()).unwrap_or(usize::MAX),
                Vec::new(),
            ));
        }
        bank.begin_turn()?;
        Ok(())
    }

    /// Inside a turn (between two evaluated steps): the turn start's
    /// re-admission again, for a host that changed since. A changed
    /// admission switches the bank and the wired lease. `true` when it
    /// switched; `false` outside a turn or without an admission.
    pub fn recheck(&self, bank: &KqBank, dense: u64, kv: u64) -> Result<bool> {
        let mut t = self.turn();
        let (Some(limits), Some(cur)) = (&self.limits, t.admission) else {
            return Ok(false);
        };
        if t.wired.is_none() {
            return Ok(false);
        }
        let (h, ws) = host_reading(self.max_memory)?;
        let o = own(bank, dense, kv)?;
        let next = match readmit_with(
            &self.profile,
            &cur,
            &self.size,
            self.rows,
            limits,
            &h,
            &o,
            ws,
        )? {
            Turn::Keep if !t.lost => return Ok(false),
            Turn::Keep => cur,
            Turn::StagingOnly(a) | Turn::Rebuild(a) => a,
        };
        crate::array::synchronize();
        self.switch(&mut t, bank, &next)?;
        tracing::info!("hy_v4_flash re-admission inside a turn: {next}");
        t.admission = Some(next);
        t.wired = Some(WiredLimitContext::bounded(
            usize::try_from(next.wired_limit()).unwrap_or(usize::MAX),
            Vec::new(),
        ));
        Ok(true)
    }

    /// Turn end (after the turn's last eval): release the wired lease, make
    /// the bank volatile.
    pub fn end_turn(&self, bank: &KqBank) -> Result<()> {
        let mut t = self.turn();
        crate::array::synchronize();
        t.wired = None;
        bank.end_turn()
    }
}

/// Engram rows through a table the store keeps for its read counters.
struct TableRows(Arc<F8Table>);

impl EngramRows for TableRows {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>> {
        self.0.rows(layer, ids)
    }
}

pub struct Store {
    pub model: Model,
    pub bank: Arc<KqBank>,
    pub table: Arc<F8Table>,
    /// GGUF bytes of the resident tensors read at load.
    pub dense_read: u64,
    /// Bytes of the resident arrays.
    pub dense_bytes: u64,
    pub sizes: Sizes,
    pub size: ModelSize,
    pub profile: Profile,
    /// Tokens one session may hold.
    pub ctx: usize,
    /// Rows of one prefill chunk.
    pub chunk: usize,
    /// Sequences the admission charged.
    pub lanes: usize,
    /// Rows a decode wave may hold (`row_cap`); below 2 no wave fits.
    pub row_cap: usize,
    /// The MTP layer is charged.
    pub mtp: bool,
    /// Seconds to allocate the bank at load (no page touched yet).
    pub bank_alloc_s: f64,
    /// The load option `maxMemoryBytes`.
    pub max_memory: Option<u64>,
    pub lease: Lease,
}

impl Store {
    /// `paths` are the shards in order (shard 0 holds the metadata).
    pub fn open(paths: &[std::path::PathBuf], opts: &Options) -> Result<Self> {
        let scan = Scan::open(paths)?;
        let sizes = Sizes::of(&scan)?;
        let mtp = opts.mtp && scan.mtp.is_some();
        let max_ctx = scan.cfg.max_context(opts.max_context)?;
        let row_cap = row_cap(
            &routing_arch()?,
            splitk_min_m(std::env::var(SPLITK_ENV).ok().as_deref()),
            &q8_shapes(&scan, mtp)?,
        )?;
        let want = if super::sched::paged_mode(false, row_cap) {
            super::sched::lanes(opts.lanes, row_cap)
        } else {
            1
        };
        let one_lane = profile(kv_per_token(&scan.cfg, mtp));
        let base = sizes.model(mtp, max_ctx);
        let limits = match &opts.plan {
            CachePlan::Admit => Some(Limits::from_env_with(&one_lane)?),
            #[cfg(test)]
            CachePlan::AdmitWith(l) => Some(*l),
            #[cfg(test)]
            CachePlan::Off | CachePlan::Slots { .. } => None,
        };
        let smallest = CHUNKS[CHUNKS.len() - 1];
        let (lanes, ctx, admitted, cache, staging, admission) = match (&limits, &opts.plan) {
            (Some(l), _) => {
                let (h, ws) = host_reading(opts.max_memory)?;
                // nothing of this model exists yet: every byte in use is others'
                let own = Own::default();
                let one = pick(&one_lane, &base, max_ctx, l, &h, &own, ws)?;
                let at = ModelSize {
                    ctx: one.ctx as u64,
                    ..base
                };
                let floor = lane_bytes(&scan.cfg, mtp, one.ctx);
                let (lanes, p) = lanes_at(one, want, |n| {
                    let lp = lane_profile(&one_lane, n, floor);
                    admit_with(&lp, &at, one.chunk as u64, l, &h, &own, ws)
                });
                tracing::info!(
                    "hy_v4_flash admission at load: {lanes} lanes, context {}, chunk {}, \
                     wave rows {row_cap}: {}",
                    p.ctx,
                    p.chunk,
                    p.admission
                );
                let a = p.admission;
                let cache = sizes.classes.split(a.cache_bytes());
                (
                    lanes,
                    p.ctx,
                    p.chunk,
                    cache,
                    a.staging_layers as usize,
                    Some(a),
                )
            }
            #[cfg(test)]
            (None, CachePlan::Slots { cache, staging }) => {
                (want, max_ctx, smallest.min(max_ctx), *cache, *staging, None)
            }
            _ => (want, max_ctx, smallest.min(max_ctx), [0, 0], 1, None),
        };
        let profile = lane_profile(&one_lane, lanes, lane_bytes(&scan.cfg, mtp, ctx));
        // the re-admission charges the rows the load admitted, the env only
        // lowers the chunk the driver runs
        let rows = admitted as u64;
        let chunk = opts.chunk_cap.map_or(admitted, |c| admitted.min(c));
        let size = ModelSize {
            ctx: ctx as u64,
            ..base
        };
        let shards = Arc::new(ShardSet::open(&scan.paths)?);
        let t_bank = Instant::now();
        let bank = Arc::new(KqBank::new(
            IoPool::new(Arc::clone(&shards), DEFAULT_THREADS)?,
            &scan.cfg.moe,
            scan.trunk_experts(),
            cache,
            staging,
        )?);
        let bank_alloc_s = t_bank.elapsed().as_secs_f64();
        let built = (|| -> Result<_> {
            let resident = Resident::load(&scan, &shards)?;
            let (read, bytes) = (resident.read, resident.bytes);
            let table = Arc::new(F8Table::open(&scan)?);
            let b = Arc::clone(&bank);
            let model = Model::new(
                scan.cfg.clone(),
                Box::new(resident),
                move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
                Box::new(TableRows(Arc::clone(&table))),
            )?;
            Ok((model, table, read, bytes))
        })();
        let (model, table, dense_read, dense_bytes) = match built {
            Ok(v) => v,
            Err(e) => {
                // the partial model is gone; keep its buffers out of MLX's free pool
                drop(bank);
                crate::array::clear_cache();
                return Err(e);
            }
        };
        let lease = Lease::new(profile, size, rows, sizes.classes, limits, admission, &bank)
            .with_max_memory(opts.max_memory);
        Ok(Self {
            model,
            bank,
            table,
            dense_read,
            dense_bytes,
            sizes,
            size,
            profile,
            ctx,
            chunk,
            lanes,
            row_cap,
            mtp,
            bank_alloc_s,
            max_memory: opts.max_memory,
            lease,
        })
    }

    pub fn admission(&self) -> Option<Admission> {
        self.lease.admission()
    }

    /// The memory governor's limit for this model: the working set every
    /// admission fills at most.
    pub fn hard_budget(&self) -> u64 {
        capped_ws(working_set(), self.max_memory)
    }

    pub fn cache_slots(&self) -> [usize; 2] {
        self.bank.cache_slots()
    }

    /// `kv`: bytes of every live session (`Session::nbytes`).
    pub fn own(&self, kv: u64) -> Result<Own> {
        own(&self.bank, self.dense_bytes, kv)
    }

    /// `kv`: bytes of every live session (`Session::nbytes`).
    pub fn begin_turn(&self, kv: u64) -> Result<()> {
        self.lease.begin_turn(&self.bank, self.dense_bytes, kv)
    }

    pub fn end_turn(&self) -> Result<()> {
        self.lease.end_turn(&self.bank)
    }

    /// `kv`: bytes of every live session (`Session::nbytes`).
    pub fn recheck(&self, kv: u64) -> Result<bool> {
        self.lease.recheck(&self.bank, self.dense_bytes, kv)
    }
}

#[cfg(test)]
mod tests;
