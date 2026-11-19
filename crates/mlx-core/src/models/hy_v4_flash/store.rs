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
//! A turn is `begin_turn` .. `end_turn` (`Lease`): the start re-admits at the
//! same context and chunk and may drop the bank to staging-only or rebuild
//! it, then holds a wired-limit lease of fixed + cache bytes until the end,
//! when the bank becomes volatile.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use napi::{Error, Result};

use super::bank::{CLASSES, ClassLayout, KqBank, slot_bytes};
use super::config::{Attn, HyV4FlashConfig};
use super::model::Model;
use super::moe::ExpertSource;
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
const CHUNK_ALIGN: usize = 256;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
/// RAM ladder entries (window clones and tails).
const LADDER_BYTES: u64 = 150 * MIB;
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
                        chunk,
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
                chunk: smallest,
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
    pub plan: CachePlan,
}

impl Options {
    pub fn from_env(max_context: Option<usize>) -> Result<Self> {
        let mtp = std::env::var(MTP_ENV).ok().is_none_or(|v| v.trim() != "0");
        Ok(Self {
            max_context,
            mtp,
            chunk_cap: chunk_cap(std::env::var(CHUNK_ENV).ok().as_deref())?,
            plan: CachePlan::Admit,
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

/// Host memory and the Metal working set for an admission.
fn host_reading() -> Result<(HostMemory, u64)> {
    #[cfg(test)]
    if let Some(r) = TEST_HOST.with(std::cell::Cell::get) {
        return Ok(r);
    }
    let h = host_memory().ok_or_else(|| bad("no host memory reading for admission"))?;
    Ok((h, working_set()))
}

/// This model's bytes inside `host_used`: resident arrays + the bank's
/// resident pages (volatile between turns). KV, scratch and the shared pool
/// stay in `others` (conservative).
pub fn own(bank: &KqBank, dense: u64) -> Result<Own> {
    let resident = bank.resident_bytes()?;
    Ok(Own {
        active: dense + resident,
        pool: 0,
        volatile: if bank.in_turn() { 0 } else { resident },
    })
}

struct TurnState {
    admission: Option<Admission>,
    wired: Option<WiredLimitContext>,
    guard: CacheLimitGuard,
    /// Bank switches: (cache slots, seconds to free the old bank and allocate).
    switches: Vec<(usize, f64)>,
}

/// The per-turn re-admission and wired-limit lease of one bank.
pub struct Lease {
    profile: Profile,
    size: ModelSize,
    rows: u64,
    classes: Classes,
    /// `None`: no admission (tests).
    limits: Option<Limits>,
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
            turn: Mutex::new(TurnState {
                admission,
                wired: None,
                guard: coordinator().register_with_cache_limit(bank.nbytes(), Some(POOL_LIMIT)),
                switches: Vec::new(),
            }),
        }
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
        bank.rebuild(slots, a.staging_layers as usize)?;
        t.guard = coordinator().register_with_cache_limit(bank.nbytes(), Some(POOL_LIMIT));
        let s = t0.elapsed().as_secs_f64();
        let n = slots[0] + slots[1];
        tracing::info!("hy_v4_flash bank switch to {slots:?} cache slots in {s:.2} s");
        t.switches.push((n, s));
        Ok(())
    }

    /// Turn start: make the bank non-volatile, re-admit, switch the bank if
    /// the admission says so, take the wired-limit lease. Errs (bank volatile
    /// again) when the fixed working set alone no longer fits.
    pub fn begin_turn(&self, bank: &KqBank, dense: u64) -> Result<()> {
        let mut t = self.turn();
        if let Some(limits) = &self.limits
            && t.wired.is_none()
        {
            let cur = t.admission.ok_or_else(|| bad("no admission"))?;
            // non-volatile first: its resident pages are then in `host_used`
            // and counted as own
            bank.begin_turn()?;
            let next = (|| {
                let (h, ws) = host_reading()?;
                let o = own(bank, dense)?;
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
                    Turn::Keep => Ok(cur),
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
    /// The MTP layer is charged.
    pub mtp: bool,
    /// Seconds to allocate the bank at load (no page touched yet).
    pub bank_alloc_s: f64,
    pub lease: Lease,
}

impl Store {
    /// `paths` are the shards in order (shard 0 holds the metadata).
    pub fn open(paths: &[std::path::PathBuf], opts: &Options) -> Result<Self> {
        let scan = Scan::open(paths)?;
        let sizes = Sizes::of(&scan)?;
        let mtp = opts.mtp && scan.mtp.is_some();
        let max_ctx = scan.cfg.max_context(opts.max_context)?;
        let profile = profile(kv_per_token(&scan.cfg, mtp));
        let base = sizes.model(mtp, max_ctx);
        let limits = match &opts.plan {
            CachePlan::Admit => Some(Limits::from_env_with(&profile)?),
            #[cfg(test)]
            CachePlan::AdmitWith(l) => Some(*l),
            #[cfg(test)]
            CachePlan::Off | CachePlan::Slots { .. } => None,
        };
        let smallest = CHUNKS[CHUNKS.len() - 1];
        let (ctx, admitted, cache, staging, admission) = match (&limits, &opts.plan) {
            (Some(l), _) => {
                let (h, ws) = host_reading()?;
                // nothing of this model exists yet: every byte in use is others'
                let p = pick(&profile, &base, max_ctx, l, &h, &Own::default(), ws)?;
                tracing::info!(
                    "hy_v4_flash admission at load: context {}, chunk {}: {}",
                    p.ctx,
                    p.chunk,
                    p.admission
                );
                let a = p.admission;
                let cache = sizes.classes.split(a.cache_bytes());
                (p.ctx, p.chunk, cache, a.staging_layers as usize, Some(a))
            }
            #[cfg(test)]
            (None, CachePlan::Slots { cache, staging }) => {
                (max_ctx, smallest, *cache, *staging, None)
            }
            _ => (max_ctx, smallest, [0, 0], 1, None),
        };
        // the re-admission charges the rows the load admitted, the env only
        // lowers the chunk the driver runs
        let rows = admitted.min(ctx) as u64;
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
        let lease = Lease::new(profile, size, rows, sizes.classes, limits, admission, &bank);
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
            mtp,
            bank_alloc_s,
            lease,
        })
    }

    pub fn admission(&self) -> Option<Admission> {
        self.lease.admission()
    }

    pub fn cache_slots(&self) -> [usize; 2] {
        self.bank.cache_slots()
    }

    pub fn own(&self) -> Result<Own> {
        own(&self.bank, self.dense_bytes)
    }

    pub fn begin_turn(&self) -> Result<()> {
        self.lease.begin_turn(&self.bank, self.dense_bytes)
    }

    pub fn end_turn(&self) -> Result<()> {
        self.lease.end_turn(&self.bank)
    }
}

#[cfg(test)]
mod tests;
