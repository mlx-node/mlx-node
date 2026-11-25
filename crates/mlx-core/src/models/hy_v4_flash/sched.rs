//! The paged mode: Hy4 on the shared continuous-batching scheduler, as Flash
//! Next (`qwen4_exp/paged.rs`), with no `ScheduledDraftVerify`.
//!
//! ```text
//! load  ─► MLX_SERVE_FORCE_SERIAL=1 or row_cap < 2 ─► flat mode (whole turn, `chat.rs`)
//!       └► else paged mode: pool of the indexed layers + owner cells, LANES = min(lanes, row_cap)
//! open  ─► busy period: first open = bank begin_turn + wired lease; later opens reset the seed
//!          quota at the next safe point; last close = end_turn (promote, volatile, lease off)
//! step  ─► safe point (promote staged copies, seed reset, readmit every READMIT_EVERY steps)
//!       ─► wave: record rows, slot maps, `Model::wave` (Q8 fused, cores per owner), MTP lag
//! slice ─► one slice per step beside the decode rows, outside the engine's row budget,
//!          ending at the turn's grid ends (C multiples + state rows); a state row of a
//!          cold-equivalent turn registers its blocks and publishes the owner's state
//! fail  ─► a step past the window slack restores the cell's mark; a shorter one truncates
//! admit ─► live cell continues │ RAM state + cached chain resumes │ cold
//! preempt ─► blocks back unpublished, cell dropped; the replay resumes from a stored state
//! close ─► the turn state T (newest block row a decode step held) publishes under write-path keys
//! disk  ─► admission reads the deepest stored state of the prompt back (the adapter's cold walk
//!          restores its blocks); every drive pass and decode step runs the tier's pass
//! MTP   ─► reserve (decode row of each lane) ─► propose (drafts in lane order, up to
//!          row_cap, the last lanes cut) ─► verify: one wave, every lane's rows ─► commit:
//!          pool rows and cell cut to start + keep (the wave already wrote the MTP rows)
//! ```
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::time::Instant;

use rand::Rng;

use napi::{Error, Result};

use super::attention::PagedPool;
use super::batch::{Core, MIN_Q8_ROWS, Owners as WaveOwners, Wave};
use super::chat::{Inner, PREFILL_CANCELLED};
use super::indexer::PagedIndexing;
use super::ladder::{Kind, RAM_STATES};
use super::model::{Logits, Model, Output, Want};
use super::moe::RowKind;
use super::mtp::{MtpLayer, lag_chunk};
use super::owner::{CellLayer, OwnerCell, Owners};
use super::paged::{BLOCK, PageGeometry, SlotMap, create};
use super::persist::{Disk, FlatTier, PoolBlocks, Read, SEGMENT_BOUND};
use super::prefill::{BLOCK_ROWS, Grid};
use super::prefix_state::{Key, States, Turn, prompt_keys, turn_keys};
use super::spec::{self, LOOKAHEAD_MAX};
use super::stats::{self, Counters, WaveSample};
use super::store::{SPLITK_ENV, Store, kv_per_token, ladder_bytes_for, lane_bytes, splitk_min_m};
use crate::array::{DType, MxArray};
use crate::cache_limit::{PoolCacheLimitGuard, coordinator};
use crate::engine::backend::{DsparkProposal, PagedBackend, PagedPrefix, PagedSpeculativeArgs};
use crate::engine::governor::governor::{
    Governor, MemoryAccounts, ModelGuard, ModelRegistration, start_under_growth_lock,
};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerState, HybridStepExecutor, NoRestoreTicket,
    ScheduledProposalRequest, ScheduledStateCapacity, ScheduledVerifyCommit, ScheduledVerifyRow,
    SchedulerOwnerContext, StepClaims, scheduler_max_num_seqs_for,
};
use crate::engine::mtp_turn::{MtpTurnArgs, run_mtp_turn, turn_lookahead_rows};
use crate::engine::paged_epilogue::FinalTokenPolicy;
use crate::engine::paged_stepper::{EvalPolicy, PagedStepModel, PagedStepper};
use crate::engine::params::ChatParams;
use crate::engine::scheduler::{PreemptionMode, StepKind, StepPlan};
use crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes;
use crate::stream::{Stream, StreamContext};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

/// Steps of one busy period between two re-admission checks.
pub const READMIT_EVERY: u32 = 128;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 scheduler: {msg}"))
}

fn no_pool() -> Error {
    bad("the load runs in flat mode: no paged pool")
}

fn no_mtp() -> Error {
    bad("the load has no MTP layer")
}

/// How an admission of `len` tokens looks the pool up: never without a
/// stored state at `row` (a KV-only cold prefix has no state to resume
/// with), and never past its row: `(skip_lookup, max_hit)`.
pub fn lookup_bounds(row: Option<usize>, len: usize) -> (bool, u32) {
    let max = row.unwrap_or(len.saturating_sub(1));
    (row.is_none(), u32::try_from(max).unwrap_or(u32::MAX))
}

/// Whether a load runs in paged mode: not forced serial and a wave row cap
/// that holds a padded Q8 call.
pub fn paged_mode(serial: bool, row_cap: usize) -> bool {
    !serial && row_cap >= MIN_Q8_ROWS as usize
}

/// Sequences the scheduler runs at once: the admitted lanes, at most the
/// decode rows one wave holds.
pub fn lanes(admitted: usize, row_cap: usize) -> usize {
    admitted.min(row_cap).max(1)
}

/// RAM prefix-state capacity for a paged load of `lanes` concurrent lanes:
/// each lane gets `RAM_STATES` slots so multi-lane workloads do not evict
/// each other's un-landed states before `DELAY`.
pub fn paged_ram_states(lanes: usize) -> usize {
    RAM_STATES.saturating_mul(lanes.max(1))
}

/// `maxConcurrentSequences`: 1 under `MLX_SERVE_FORCE_SERIAL`, else the
/// engine's cap over `lanes`.
pub fn napi_concurrency(serial: bool, lanes: usize) -> u32 {
    if serial {
        1
    } else {
        u32::try_from(scheduler_max_num_seqs_for(lanes)).unwrap_or(u32::MAX)
    }
}

/// Prefill slice ends of `prompt` past `cached`: the grid's chunk ends, the
/// prompt's state rows among them; `None` when `prompt_tokens` is not the
/// prompt's length.
pub fn prefill_ends(
    grid: &Grid,
    prompt: &[u32],
    prompt_tokens: u32,
    cached: u32,
) -> Option<Vec<u32>> {
    if prompt.len() != prompt_tokens as usize {
        return None;
    }
    grid.ends(cached as usize, prompt)
        .into_iter()
        .map(|e| u32::try_from(e).ok())
        .collect()
}

/// The engine's pinned slice ends of a prompt of `prompt_tokens` past
/// `cached`: `prompt`'s grid ends when it is that prompt, else every
/// multiple of `grid`'s chunk.
pub fn slice_breaks(
    grid: &Grid,
    prompt: Option<&[u32]>,
    prompt_tokens: u32,
    cached: u32,
) -> Vec<u32> {
    if let Some(ends) = prompt.and_then(|p| prefill_ends(grid, p, prompt_tokens, cached)) {
        return ends;
    }
    let chunk = grid.chunk().max(1) as u64;
    let (from, to) = (u64::from(cached), u64::from(prompt_tokens));
    (from / chunk + 1..)
        .map(|k| k * chunk)
        .take_while(|&e| e < to)
        .filter_map(|e| u32::try_from(e).ok())
        .collect()
}

/// Per-block extra keys of `tokens` chunked at `ends` (from row 0): block
/// `i` carries the hash of the chunk that wrote it (its tokens, its start
/// and end, the grid and the ends inside it), so a cached block matches
/// only one written by the same chunk over the same tokens. Blocks of the
/// last chunk get none. Flash Next's `pass_keys`.
pub fn pass_keys(tokens: &[u32], ends: &[usize], grid: usize) -> Vec<Vec<u64>> {
    let Some(&last) = ends.last() else {
        return Vec::new();
    };
    let mut keys = Vec::new();
    let mut before = 0;
    let mut after = 0;
    let mut memo: Option<(usize, usize, u64)> = None;
    for block in 0.. {
        let (low, high) = (block * BLOCK_ROWS, (block + 1) * BLOCK_ROWS);
        while ends.get(before).is_some_and(|&end| end <= low) {
            before += 1;
        }
        while ends.get(after).is_some_and(|&end| end < high) {
            after += 1;
        }
        let Some(&end) = ends.get(after) else {
            break;
        };
        if end == last {
            break;
        }
        let start = before.checked_sub(1).map_or(0, |i| ends[i]);
        let key = match memo {
            Some((s, e, key)) if s == start && e == end => key,
            _ => {
                let mut extra = vec![grid as u64, start as u64, end as u64];
                extra.extend(
                    ends[before..after]
                        .iter()
                        .filter(|&&row| row > start && row < end)
                        .map(|&row| row as u64),
                );
                let key = mlx_paged_attn::hash_tokens(&tokens[start..end], 0, &extra);
                memo = Some((start, end, key));
                key
            }
        };
        keys.push(vec![key]);
    }
    keys
}

/// What a turn edge does to the busy period.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// The first open turn: the period begins.
    Begin,
    /// A turn joined a running period.
    Join,
    /// The last turn closed: the period ends.
    End,
    None,
}

/// The turns open on the scheduler and the exclusive lane: one busy period
/// from the first open to the last close.
#[derive(Debug, Default)]
pub struct Busy {
    open: HashSet<SeqId>,
    exclusive: bool,
    live: bool,
}

impl Busy {
    pub fn is_live(&self) -> bool {
        self.live
    }

    /// A scheduled turn other than `seq` is open.
    pub fn others_open(&self, seq: SeqId) -> bool {
        self.open.iter().any(|&s| s != seq)
    }

    fn join_or_begin(&mut self) -> Edge {
        if self.live {
            Edge::Join
        } else {
            self.live = true;
            Edge::Begin
        }
    }

    fn settle(&mut self) -> Edge {
        if self.live && self.open.is_empty() && !self.exclusive {
            self.live = false;
            Edge::End
        } else {
            Edge::None
        }
    }

    /// The scheduler admitted `seq`.
    pub fn opened(&mut self, seq: SeqId) -> Edge {
        if self.open.insert(seq) {
            self.join_or_begin()
        } else {
            Edge::None
        }
    }

    /// `seq`'s scheduled turn ended.
    pub fn closed(&mut self, seq: SeqId) -> Edge {
        self.open.remove(&seq);
        self.settle()
    }

    /// A turn's cancel flag was set (`Some`) or cleared: with no scheduled
    /// turn open, a set flag is an exclusive turn.
    pub fn flag(&mut self, set: bool) -> Edge {
        if !set {
            self.exclusive = false;
            return self.settle();
        }
        if self.open.is_empty() && !self.exclusive {
            self.exclusive = true;
            return self.join_or_begin();
        }
        Edge::None
    }
}

/// What a safe point runs before the next step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Due {
    pub seed_reset: bool,
    pub readmit: bool,
}

/// The work due at the start of each step of a busy period: a seed reset
/// once a turn joined, a re-admission every `every` steps.
#[derive(Debug)]
pub struct SafePoint {
    seed_reset: bool,
    steps: u32,
    every: u32,
}

impl SafePoint {
    pub fn new(every: u32) -> Self {
        Self {
            seed_reset: false,
            steps: 0,
            every: every.max(1),
        }
    }

    /// A busy period began (the bank's turn start reset the seeds).
    pub fn begun(&mut self) {
        self.seed_reset = false;
        self.steps = 0;
    }

    /// A turn joined the running period.
    pub fn joined(&mut self) {
        self.seed_reset = true;
    }

    /// The next step starts.
    pub fn step(&mut self) -> Due {
        self.steps = self.steps.wrapping_add(1);
        Due {
            seed_reset: std::mem::take(&mut self.seed_reset),
            readmit: self.steps.is_multiple_of(self.every),
        }
    }
}

/// The pool operations the hooks take, so their order is testable without
/// Metal.
pub trait Requests {
    /// Whether `seq` holds a request.
    fn holds(&self, seq: SeqId) -> bool;
    fn release(&mut self, seq: SeqId) -> Result<()>;
    /// Registers `seq`'s full blocks below `upto` in the prefix cache.
    fn register(&mut self, seq: SeqId, upto: usize, keys: &[Vec<u64>], salt: u64) -> Result<()>;
}

impl Requests for PagedKVCacheAdapter {
    fn holds(&self, seq: SeqId) -> bool {
        self.block_table_for(seq).is_some()
    }

    fn release(&mut self, seq: SeqId) -> Result<()> {
        self.release_request_for(seq).map(|_| ()).map_err(bad)
    }

    fn register(&mut self, seq: SeqId, upto: usize, keys: &[Vec<u64>], salt: u64) -> Result<()> {
        self.activate_request(seq).map_err(bad)?;
        self.register_committed_blocks(upto, keys, salt)
            .map_err(bad)
    }
}

/// A preempted turn's blocks go back without publication: its rows past
/// the last stored state include decode rows, which are not what a cold
/// prefill writes.
pub fn preempt<R: Requests>(pool: &mut R, seq: SeqId) -> Result<()> {
    if pool.holds(seq) {
        pool.release(seq)?;
    }
    Ok(())
}

/// One admitted turn's prompt and how it publishes.
pub struct TurnPlan {
    pub prompt: Vec<u32>,
    /// Per-block extra keys of the prompt as this turn writes it
    /// (`prompt_keys`).
    pub keys: Vec<Vec<u64>>,
    pub salt: u64,
    /// Its state rows and its turn state publish.
    pub publish: bool,
    pub turn: Turn,
    /// The write-path keys' seed of this turn.
    pub nonce: u64,
}

/// A refused re-admission inside a busy period `end`s its lease and fails
/// the step; the next step starts the period over through the turn start's
/// admission.
pub fn readmit(checked: Result<bool>, end: impl FnOnce()) -> Result<()> {
    match checked {
        Ok(_) => Ok(()),
        Err(e) => {
            tracing::warn!("hy_v4_flash re-admission refused: {}", e.reason);
            end();
            Err(e)
        }
    }
}

/// The rows of one verify wave: one decode row per lane that reserved,
/// then drafts in proposal order while rows are left under `row_cap`, so
/// the last lanes lose their drafts first. A reserve after a proposal
/// starts the next wave.
#[derive(Debug, Default)]
pub struct WaveBudget {
    lanes: HashSet<SeqId>,
    drafts: usize,
    proposing: bool,
    /// Drafts each decode row of the planned step may reserve pool rows
    /// for: every anchor first, then drafts in plan order under the row cap.
    allow: HashMap<SeqId, usize>,
    /// Rows each lane of the wave reserved.
    held: HashMap<SeqId, usize>,
}

impl WaveBudget {
    /// The planned step's decode rows `(seq, tokens)` under `row_cap`.
    pub fn plan(&mut self, rows: &[(SeqId, usize)], row_cap: usize) {
        self.allow.clear();
        let mut left = row_cap.saturating_sub(rows.len());
        for &(seq, tokens) in rows {
            let n = tokens.saturating_sub(1).min(left);
            left -= n;
            self.allow.insert(seq, n);
        }
    }

    /// Rows `seq` reserves of the `queries` it asks: its anchor and the
    /// drafts its plan allows.
    pub fn rows_for(&self, seq: SeqId, queries: usize) -> usize {
        self.allow
            .get(&seq)
            .map_or(queries, |&n| queries.min(n + 1))
    }

    /// `seq` holds a decode row of the wave being planned.
    pub fn reserved(&mut self, seq: SeqId) {
        if self.proposing {
            self.close();
        }
        self.lanes.insert(seq);
    }

    /// `seq` reserved `rows` pool rows for the wave being planned.
    pub fn reserved_rows(&mut self, seq: SeqId, rows: usize) {
        self.reserved(seq);
        self.held.insert(seq, rows);
    }

    /// Drafts `seq` may propose, at most `cap` and the rows it reserved.
    pub fn grant(&mut self, seq: SeqId, cap: usize, row_cap: usize) -> usize {
        self.proposing = true;
        self.lanes.insert(seq);
        let cap = self
            .held
            .get(&seq)
            .map_or(cap, |&rows| cap.min(rows.saturating_sub(1)));
        let left = row_cap.saturating_sub(self.lanes.len() + self.drafts);
        let n = cap.min(left);
        self.drafts += n;
        n
    }

    /// Rows the wave holds so far.
    pub fn rows(&self) -> usize {
        self.lanes.len() + self.drafts
    }

    /// The wave ran (or was dropped).
    pub fn close(&mut self) {
        self.lanes.clear();
        self.held.clear();
        self.drafts = 0;
        self.proposing = false;
    }
}

/// One owner's verify rows, held in the pool and its cell until the commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Open {
    pub start: usize,
    pub rows: usize,
}

/// The cell length a commit keeping `keep` rows of `open` leaves.
pub fn commit_len(open: Open, keep: usize) -> Result<usize> {
    if keep > open.rows {
        return Err(bad(format!(
            "a commit of {keep} rows of a verify of {}",
            open.rows
        )));
    }
    Ok(open.start + keep)
}

/// Effective decode-wave row cap in paged mode. With MTP enabled and no
/// explicit split-K override, accommodates all admitted lanes at maximum draft
/// depth (`lanes * LOOKAHEAD_MAX`) so multi-lane depth 2-3 waves are not
/// throttled back to depth 1.
pub fn mtp_wave_row_cap(
    base_row_cap: usize,
    lanes: usize,
    mtp: bool,
    splitk_override: bool,
) -> usize {
    if mtp && !splitk_override && base_row_cap >= MIN_Q8_ROWS as usize {
        base_row_cap.max(lanes.saturating_mul(LOOKAHEAD_MAX))
    } else {
        base_row_cap
    }
}

/// Whether `rows` more rows fit an owner of `len` positions: one cut must
/// be able to drop all but the anchor, one wave must hold them, and the
/// context must.
pub fn rows_fit(rows: usize, len: usize, ctx: usize, row_cap: usize) -> bool {
    rows > 0 && rows <= LOOKAHEAD_MAX.min(row_cap) && len + rows <= ctx
}

/// The paged mode's MTP: the verify being planned and the open ones.
#[derive(Debug, Default)]
pub struct Spec {
    pub budget: WaveBudget,
    pub open: HashMap<SeqId, Open>,
}

/// The paged mode of a load.
pub struct Paged {
    pub adapter: PagedKVCacheAdapter,
    pub owners: Owners,
    /// The active owner's committed history.
    pub saved: Vec<u32>,
    pub turns: HashMap<SeqId, TurnPlan>,
    pub states: States,
    pub busy: Busy,
    pub safe: SafePoint,
    pub lanes: usize,
    pub row_cap: usize,
    pub published: u64,
    pub restored: u64,
    pub spec: Spec,
    /// The stored states' disk tier; `None` when it is off.
    pub(crate) disk: Option<Disk>,
    /// Each owner's keys of its cell's whole blocks, as its turns wrote them.
    pub lines: HashMap<SeqId, Vec<Vec<u64>>>,
    _pool: PoolCacheLimitGuard,
    /// The model's registration with the memory governor.
    pub(crate) governor: ModelGuard<'static>,
}

impl Drop for Paged {
    /// A dispose writes the waiting states and drains the copies within one
    /// bound before the tier closes.
    fn drop(&mut self) {
        let Paged {
            disk,
            states,
            adapter,
            ..
        } = self;
        if let Some(disk) = disk.as_mut() {
            disk.dispose(states.store_mut(), &mut PoolBlocks(adapter));
        }
    }
}

impl Paged {
    pub(crate) fn model_registration(store: &Store, page: u64) -> ModelRegistration {
        ModelRegistration {
            model_id: crate::engine::compiled_lock::QWEN35_MODEL_ID_COUNTER
                .fetch_add(1, Ordering::Relaxed),
            hard_budget: store.hard_budget(),
            untracked: store.governor_untracked_bytes(),
            page_bytes: page,
        }
    }

    /// The pool for `store`'s admitted context (one partial block per lane
    /// on top), allocated whole at load.
    /// The model registers with the memory governor, which admits the pool
    /// as the start's runway.
    pub fn open(store: &Store) -> Result<Self> {
        let cfg = store.model.cfg();
        let g = PageGeometry::of(cfg)?;
        let lanes = lanes(store.lanes, store.row_cap);
        let splitk_override = splitk_min_m(std::env::var(SPLITK_ENV).ok().as_deref()).is_some();
        let row_cap = mtp_wave_row_cap(store.row_cap, lanes, store.mtp, splitk_override);
        let max_blocks = store.ctx.div_ceil(BLOCK as usize) + lanes + 1;
        let max_blocks = u32::try_from(max_blocks).map_err(|_| bad("pool blocks exceed u32"))?;
        let page = g.block_bytes()?;
        let initial_blocks = if store.is_direct() {
            crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(page)
                .max(u32::try_from(lanes).unwrap_or(1).saturating_add(4))
                .min(max_blocks)
        } else {
            max_blocks
        };
        let registration = Self::model_registration(store, page);
        crate::array::memory::synchronize_and_clear_cache();
        let gov: &'static Governor = crate::engine::governor::governor();
        let (governor, (adapter, pool)) = start_under_growth_lock(gov, || {
            gov.load_pool(registration, u64::from(initial_blocks) * page, || {
                let mut adapter = create(&g, store.ctx, initial_blocks, max_blocks, u64::MAX)
                    .map_err(|e| e.reason)?;
                let pool = coordinator().register_pool(block_bytes(&adapter));
                if initial_blocks < max_blocks {
                    let pool_id = pool.id();
                    adapter.set_pool_growth_notifier(Some(std::sync::Arc::new(move |bytes| {
                        coordinator().update_pool(pool_id, bytes);
                    })));
                }
                Ok((adapter, pool))
            })
            .map_err(Error::from_reason)
        })?;
        Ok(Self {
            adapter,
            owners: Owners::new(),
            saved: Vec::new(),
            turns: HashMap::new(),
            states: States::with_cap(paged_ram_states(lanes)),
            busy: Busy::default(),
            safe: SafePoint::new(READMIT_EVERY),
            lanes,
            row_cap,
            published: 0,
            restored: 0,
            spec: Spec::default(),
            disk: None,
            lines: HashMap::new(),
            _pool: pool,
            governor,
        })
    }

    /// Hands the pool's blocks and the RAM states to `disk` and takes back
    /// what a persistent tier holds.
    pub(crate) fn attach_disk(&mut self, disk: Disk) -> Result<u64> {
        disk.attach(&mut self.adapter)?;
        let mut disk = disk;
        let adopted = disk.adopt(self.states.store_mut());
        self.disk = Some(disk);
        Ok(adopted)
    }

    /// The disk's landings, and at most every few steps its due states and
    /// block copies. True when a write landed.
    pub fn top_up_disk(&mut self) -> bool {
        let Paged {
            disk,
            states,
            adapter,
            ..
        } = self;
        let Some(disk) = disk.as_mut() else {
            return false;
        };
        disk.drops(states.dropped());
        disk.step(states.store_mut(), &mut PoolBlocks(adapter))
    }

    /// `[states with a RAM copy, their bytes, published, restored]`.
    pub fn counts(&self) -> [u64; 4] {
        [
            self.states.ram_len() as u64,
            self.states.held_bytes(),
            self.published,
            self.restored,
        ]
    }

    /// Bytes of the pool's blocks.
    pub fn pool_bytes(&self) -> u64 {
        block_bytes(&self.adapter)
    }

    /// Bytes outside the pool: owner cells, RAM states and the turn states
    /// the open turns hold.
    pub fn state_bytes(&self) -> u64 {
        let held: u64 = self.turns.values().map(|t| t.turn.held_bytes()).sum();
        self.owners.bytes() + self.states.held_bytes() + held
    }

    fn live_tokens(&self, seq: SeqId) -> &[u32] {
        self.adapter.request_tokens_for(seq).unwrap_or(&[])
    }

    /// `seq`'s pool rows and cell cut to `len`, capturing any newly committed
    /// block-row turn state; an owner that cannot go there is dropped. Returns
    /// the number of rows cut.
    pub(crate) fn cut_owner(&mut self, seq: SeqId, len: usize, mtp_loaded: bool) -> Result<usize> {
        let was = self.owners.active();
        let held = self.adapter.request_tokens_for(seq).map_or(0, <[u32]>::len);
        let cell = self.owners.get(seq).map_or(0, OwnerCell::len);
        if held == len && cell == len {
            if let Some(c) = self.owners.get_mut(seq) {
                c.drop_mark();
            }
            hold_turn_state(self, seq, mtp_loaded);
            return Ok(0);
        }
        let cut = if held != cell || len > held {
            Err(bad(format!(
                "owner {seq}: a cut to {len} of {held} pool rows and a cell of {cell}"
            )))
        } else {
            forget_above(self, seq, len);
            Self::cut_rows(self, seq, held - len, len)
        };
        match &cut {
            Ok(()) => {
                if let Some(c) = self.owners.get_mut(seq) {
                    c.drop_mark();
                }
                hold_turn_state(self, seq, mtp_loaded);
            }
            Err(_) => lose(self, seq, was == Some(seq)),
        }
        reactivate(self, was)?;
        cut.map(|()| held - len)
    }

    fn cut_rows(p: &mut Paged, seq: SeqId, back: usize, len: usize) -> Result<()> {
        let back = u32::try_from(back).map_err(|_| bad("rows exceed u32"))?;
        p.adapter.rollback_last_tokens_for(seq, back).map_err(bad)?;
        let history = p.adapter.request_tokens_for(seq).unwrap_or(&[]);
        match p.owners.get_mut(seq) {
            Some(c) => c.truncate(len, history),
            None => Err(bad(format!("owner {seq} has no cell"))),
        }
    }
}

/// Bytes of `adapter`'s blocks.
fn block_bytes(adapter: &PagedKVCacheAdapter) -> u64 {
    let blocks = adapter
        .block_telemetry()
        .map_or(0, |t| u64::from(t.total_blocks));
    blocks * adapter.bytes_per_block().unwrap_or(0)
}

/// Routed experts one row picks over the trunk's streamed layers.
fn picks_per_row(store: &Store) -> u64 {
    let cfg = store.model.cfg();
    let layers = cfg.trunk().iter().filter(|l| !l.dense_ffn).count();
    (cfg.moe.n_used * layers) as u64
}

/// A decode wave's counters at its start.
struct WaveClock {
    counters: Counters,
    evals: u64,
    t0: Instant,
}

impl WaveClock {
    fn start(store: &Store) -> Self {
        Self {
            counters: Counters::now(&store.bank, &store.table),
            evals: stats::evals(),
            t0: Instant::now(),
        }
    }

    /// The wave of `rows` into `waves`, and its line into the stats file.
    fn record(self, store: &Store, rows: &[Rows<'_>], waves: &mut stats::WaveLog) {
        let owners: Vec<(SeqId, usize)> = rows.iter().map(|r| (r.seq, r.ids.len())).collect();
        let sample = WaveSample {
            owners: &owners,
            picks_per_row: picks_per_row(store),
            counters: Counters::now(&store.bank, &store.table).since(&self.counters),
            evals: stats::evals().saturating_sub(self.evals),
            secs: self.t0.elapsed().as_secs_f64(),
        };
        waves.record(&sample);
        if std::env::var_os(stats::STATS_ENV).is_some()
            && let Err(e) = stats::emit(&sample.json())
        {
            tracing::warn!("hy_v4_flash stats: {e}");
        }
    }
}

/// An empty owner cell, with an empty draft when the MTP layer is loaded.
fn new_cell(store: &Store, mtp: Option<&MtpLayer>) -> Result<OwnerCell> {
    OwnerCell::new(store.model.cfg(), mtp.map(|m| m.new_draft(store.ctx)))
}

/// One owner's rows of a step.
pub struct Rows<'a> {
    pub seq: SeqId,
    pub ids: &'a [u32],
    pub logits: Logits,
}

/// The wave's owners: their taken cells, the pool, their slot maps.
struct PagedOwners<'a> {
    adapter: &'a mut PagedKVCacheAdapter,
    rows: &'a [Rows<'a>],
    cells: &'a mut [OwnerCell],
    starts: &'a [usize],
    maps: &'a [SlotMap],
    run_mtp: bool,
}

impl WaveOwners for PagedOwners<'_> {
    fn count(&self) -> usize {
        self.rows.len()
    }

    fn ids(&self, owner: usize) -> &[u32] {
        self.rows.get(owner).map_or(&[], |r| r.ids)
    }

    fn history(&self, owner: usize) -> &[u32] {
        let (Some(r), Some(&start)) = (self.rows.get(owner), self.starts.get(owner)) else {
            return &[];
        };
        let t = self.adapter.request_tokens_for(r.seq).unwrap_or(&[]);
        &t[..start.min(t.len())]
    }

    fn want(&self, owner: usize) -> Want {
        match (self.rows.get(owner), self.cells.get(owner)) {
            (Some(r), Some(c)) => Want {
                logits: r.logits,
                hidden: self.run_mtp && c.draft().is_some(),
            },
            _ => Want::STATE,
        }
    }

    fn is_paged(&self) -> bool {
        true
    }

    fn core(&mut self, owner: usize, call: Core<'_>) -> Result<MxArray> {
        let seq = self
            .rows
            .get(owner)
            .map(|r| r.seq)
            .ok_or_else(|| bad(format!("owner {owner} of the wave")))?;
        let map = self
            .maps
            .get(owner)
            .ok_or_else(|| bad(format!("no slot map for owner {owner}")))?;
        let l = call.blk.layer();
        let part = self
            .cells
            .get_mut(owner)
            .and_then(|c| c.layers_mut().get_mut(l))
            .ok_or_else(|| bad(format!("owner {owner} has no layer {l}")))?;
        let mla = call.blk.mla();
        match part {
            CellLayer::Window(ring) => {
                mla.core_sliding_for(&call.w.attn, call.p, call.start, ring, call.kind)
            }
            CellLayer::Index(state) => {
                let (Some(indexer), Some(weights)) = (call.blk.indexer(), call.w.indexer.as_ref())
                else {
                    return Err(bad(format!(
                        "layer {l}: an indexed cell on a layer without one"
                    )));
                };
                self.adapter.activate_request(seq).map_err(bad)?;
                let floor = u32::try_from(call.start).map_err(|_| bad("position exceeds u32"))?;
                let mut pool = PagedPool {
                    adapter: &mut *self.adapter,
                    floor,
                    slots: map,
                };
                let mut ix = PagedIndexing {
                    indexer,
                    weights,
                    state,
                    kind: call.kind,
                };
                mla.core_indexed_paged(&call.w.attn, call.p, call.start, &mut pool, &mut ix)
            }
        }
    }
}

/// One step's fixed inputs.
struct Run<'a> {
    model: &'a Model,
    mtp: Option<&'a MtpLayer>,
    kind: RowKind,
    row_cap: usize,
    cancel: Option<&'a dyn Fn(usize) -> bool>,
}

impl Run<'_> {
    /// Records every owner's ids, runs the wave and the MTP rows it closes,
    /// advances the cells. `recorded` counts the owners whose ids the pool
    /// recorded (the caller rolls them back on error).
    fn step(
        &self,
        adapter: &mut PagedKVCacheAdapter,
        rows: &[Rows<'_>],
        cells: &mut [OwnerCell],
        starts: &[usize],
        recorded: &mut usize,
    ) -> Result<Vec<Output>> {
        for (r, &start) in rows.iter().zip(starts) {
            adapter.activate_request(r.seq).map_err(bad)?;
            if adapter.request_tokens().len() != start {
                return Err(bad(format!(
                    "owner {}: the pool holds {} rows, its cell {start}",
                    r.seq,
                    adapter.request_tokens().len()
                )));
            }
            adapter.record_tokens(r.ids).map_err(bad)?;
            *recorded += 1;
        }
        let mut maps = Vec::with_capacity(rows.len());
        for r in rows {
            adapter.activate_request(r.seq).map_err(bad)?;
            maps.push(SlotMap::of(adapter)?);
        }
        let spans: Vec<(usize, usize)> = rows
            .iter()
            .zip(starts)
            .map(|(r, &s)| (s, r.ids.len()))
            .collect();
        let cap = (self.kind == RowKind::Decode).then_some(self.row_cap);
        let wave = Wave::new(self.kind, &spans, cap)?;
        let outs = self.model.wave(
            &mut PagedOwners {
                adapter: &mut *adapter,
                rows,
                cells: &mut *cells,
                starts,
                maps: &maps,
                run_mtp: self.mtp.is_some(),
            },
            &wave,
            self.cancel,
            None,
        )?;
        adapter.eval_pending_pool_writes().map_err(bad)?;
        for (cell, r) in cells.iter_mut().zip(rows) {
            cell.advance(r.ids)?;
        }
        if let Some(m) = self.mtp {
            let tp_mtp = stats::prof_start();
            let _scope = stats::DeferScope::enter();
            for ((cell, r), (out, &start)) in
                cells.iter_mut().zip(rows).zip(outs.iter().zip(starts))
            {
                let Some(d) = cell.draft_mut() else {
                    continue;
                };
                if d.len() + 1 != start.max(1) {
                    return Err(bad(format!(
                        "owner {}: {} draft rows for a step at {start}",
                        r.seq,
                        d.len()
                    )));
                }
                let h = out
                    .hidden
                    .as_ref()
                    .ok_or_else(|| bad("the wave returned no hidden rows"))?;
                lag_chunk(m, self.model, d, start, r.ids, h)?;
            }
            if self.kind.is_decode() && !stats::profiling() {
                stats::async_flush();
            } else {
                stats::flush("hy_v4_flash MTP lag")?;
            }
            stats::prof_end(stats::Phase::Mtp, tp_mtp, &[])?;
        }
        Ok(outs)
    }
}

/// The disk learns of `seq`'s state `id` at `row` (a replay state when
/// `replay`) and copies out what it owes.
fn disk_published(p: &mut Paged, seq: SeqId, row: usize, id: u64, replay: Option<u64>) {
    let Paged {
        disk,
        states,
        adapter,
        ..
    } = p;
    let Some(disk) = disk.as_mut() else {
        return;
    };
    disk.drops(states.dropped());
    let draft = states.get(id).and_then(|s| s.draft.as_ref());
    disk.published(states.store(), seq, row, draft, replay);
    disk.top_up(seq, &mut PoolBlocks(adapter));
}

/// `seq`'s lineage `tokens[..row]` keyed by `keys` owes the disk its
/// chain from now on.
fn disk_start(p: &mut Paged, seq: SeqId, tokens: &[u32], keys: &[Vec<u64>], salt: u64, row: usize) {
    let Some(disk) = p.disk.as_mut() else {
        return;
    };
    let blocks = row / BLOCK_ROWS;
    if blocks == 0 || keys.len() < blocks || tokens.len() < row {
        return;
    }
    let hot = compute_paged_prefix_block_hashes(
        tokens,
        row as u32,
        BLOCK_ROWS as u32,
        &keys[..blocks],
        salt,
    )
    .unwrap_or_default();
    disk.start(seq, &tokens[..row], keys, &hot, salt);
}

/// `seq`'s turn ended: its turn state publishes under write-path keys, its
/// blocks below registered under them, and the owner keeps the keys of its
/// cell's whole blocks for its next turn.
fn publish_turn(p: &mut Paged, seq: SeqId, plan: &mut TurnPlan) {
    let Some(kv_end) = p.owners.get(seq).map(OwnerCell::len) else {
        return;
    };
    let tokens = p.live_tokens(seq).to_vec();
    if tokens.len() != kv_end {
        return;
    }
    let keys = turn_keys(&tokens, &plan.keys, kv_end, plan.nonce);
    p.lines.insert(seq, keys.clone());
    let Some((at, state)) = plan.turn.close(kv_end) else {
        return;
    };
    let (row, salt) = (at.row, plan.salt);
    let blocks = row / BLOCK_ROWS;
    if keys.len() < blocks {
        return;
    }
    if let Err(e) = Requests::register(&mut p.adapter, seq, row, &keys[..blocks], salt) {
        tracing::warn!("hy_v4_flash turn state blocks below {row}: {}", e.reason);
        return;
    }
    let key = Key {
        tokens: &tokens[..row],
        keys: &keys[..blocks],
        salt,
    };
    match p.states.publish(key, state, Kind::Turn) {
        Ok(Some(id)) => {
            p.published += 1;
            disk_start(p, seq, &tokens, &keys, salt, row);
            disk_published(p, seq, row, id, Some(id));
        }
        Ok(None) => {}
        Err(e) => tracing::warn!("hy_v4_flash turn state at {row}: {}", e.reason),
    }
}

/// The deepest state on disk only that `prompt` resumes from past `hit`,
/// whose chain is whole, read back and given its RAM copy; two at most are
/// tried, one that does not read back going with its object. `None`
/// keeps `hit`.
fn resume_from_disk(
    p: &mut Paged,
    prompt: Key<'_>,
    hit: Option<(u64, usize)>,
) -> Option<(u64, usize)> {
    let floor = hit.map_or(0, |h| h.1);
    let found = p.states.disk_only_above(prompt, floor);
    for (id, row) in found.into_iter().take(2) {
        let (key, extra) = {
            let entry = p.states.store().get(id)?;
            (entry.disk()?.key, entry.extra_keys().to_vec())
        };
        let Paged {
            disk,
            states,
            adapter,
            ..
        } = &mut *p;
        let disk = disk.as_mut()?;
        let segments = disk.chain_whole(prompt.tokens, &extra, prompt.salt, row, &|h| {
            adapter.prefix_cached(h)
        });
        let Some(segments) = segments else {
            disk.discard(states.store_mut(), id);
            continue;
        };
        match disk.read_state(
            states.store_mut(),
            key,
            row,
            &extra,
            &segments,
            SEGMENT_BOUND,
        ) {
            Read::State(s) => {
                if !states.promote(id, s) {
                    return None;
                }
                tracing::debug!(
                    row,
                    prompt = prompt.tokens.len(),
                    "hy_v4_flash: read a stored state back from disk"
                );
                return Some((id, row));
            }
            Read::Busy => return None,
            Read::Failed => {
                states.store_mut().remove(id);
            }
        }
    }
    None
}

/// `seq` is about to be cut to `len`: its turn state past `len` goes, its
/// stored states past `len` go unless their chain stays cached, and the
/// disk stops copying blocks it no longer holds.
fn forget_above(p: &mut Paged, seq: SeqId, len: usize) {
    let Paged {
        adapter,
        states,
        turns,
        disk,
        ..
    } = p;
    if let Some(plan) = turns.get_mut(&seq) {
        plan.turn.cut(len);
        if let Some(tokens) = adapter.request_tokens_for(seq) {
            let adapter = &*adapter;
            states.cut(tokens, plan.salt, len, &|h| adapter.prefix_cached(h));
        }
    }
    if let Some(disk) = disk.as_mut() {
        disk.cut(seq, len);
    }
}

/// A publishing turn of `seq` whose cell stands at a block row keeps the
/// cell's state there as its turn state candidate.
pub(super) fn hold_turn_state(p: &mut Paged, seq: SeqId, mtp_loaded: bool) {
    let Some(plan) = p.turns.get_mut(&seq).filter(|t| t.publish) else {
        return;
    };
    let Some(cell) = p.owners.get_mut(seq) else {
        return;
    };
    if cell.is_empty() || !cell.len().is_multiple_of(BLOCK_ROWS) || !plan.turn.wants(cell.len()) {
        return;
    }
    match cell.state() {
        Ok(Some(s)) if !mtp_loaded || s.draft.is_some() => plan.turn.hold(s),
        Ok(Some(_) | None) => {}
        Err(e) => tracing::warn!("hy_v4_flash turn state of owner {seq}: {}", e.reason),
    }
}

/// Drops `seq`'s request, cell and plans after a step it cannot go back
/// from; `active` when it was the active owner.
fn lose(p: &mut Paged, seq: SeqId, active: bool) {
    tracing::warn!("hy_v4_flash: owner {seq} lost its cell in a failed step");
    let _ = p.adapter.release_request_for(seq);
    p.owners.release(seq);
    p.turns.remove(&seq);
    p.lines.remove(&seq);
    if let Some(disk) = p.disk.as_mut() {
        disk.close(seq);
    }
    p.spec.open.remove(&seq);
    if active {
        p.saved.clear();
    }
}

/// Makes `was` the active owner again in the table and the pool.
fn reactivate(p: &mut Paged, was: Option<SeqId>) -> Result<()> {
    if let Some(seq) = was
        && p.owners.contains(seq)
    {
        p.owners
            .activate(seq, || Err(bad("the active cell vanished")))?;
        p.adapter.activate_request(seq).map_err(bad)?;
    }
    Ok(())
}

fn empty_proposal() -> DsparkProposal {
    DsparkProposal {
        draft_ids: Vec::new(),
        device_draft_ids: None,
        device_draft_sparse: None,
        draft_dists: Vec::new(),
        draft_sparse_dists: Vec::new(),
        keep_probabilities: None,
    }
}

/// The flag's cancel read, per layer and per chunk.
fn cancelled(flag: &Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) -> bool {
    flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed))
}

impl Inner {
    pub(crate) fn paged(&self) -> Result<&Paged> {
        self.paged.as_ref().ok_or_else(no_pool)
    }

    pub(crate) fn paged_mut(&mut self) -> Result<&mut Paged> {
        self.paged.as_mut().ok_or_else(no_pool)
    }

    /// Runs one step over `rows` (distinct owners) with the cells out of the
    /// table. On error every owner goes back to its start; an owner whose
    /// cell cannot is released. A verify owner keeps its mark until its cut.
    /// `hold_turn` captures a block-row turn state candidate after the wave;
    /// speculative verify waves pass `false` and capture in [`Self::cut_owner`]
    /// once acceptance is decided.
    fn wave_rows(
        &mut self,
        rows: &[Rows<'_>],
        kind: RowKind,
        run_mtp: bool,
        hold_turn: bool,
        cancel: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<Vec<Output>> {
        let out = self.wave_once(rows, kind, run_mtp, hold_turn, cancel);
        if kind.is_decode() {
            self.publish(None);
        }
        out
    }

    fn wave_once(
        &mut self,
        rows: &[Rows<'_>],
        kind: RowKind,
        run_mtp: bool,
        hold_turn: bool,
        cancel: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<Vec<Output>> {
        let mut seen = HashSet::new();
        if rows.is_empty() || rows.iter().any(|r| !seen.insert(r.seq) || r.ids.is_empty()) {
            return Err(bad("a step needs distinct owners, each with rows"));
        }
        let Inner {
            store,
            mtp,
            paged,
            waves,
            ..
        } = self;
        let p = paged.as_mut().ok_or_else(no_pool)?;
        let was = p.owners.active();
        let clock = kind.is_decode().then(|| WaveClock::start(store));
        let mut fresh = Vec::with_capacity(rows.len());
        for r in rows {
            fresh.push(if p.owners.contains(r.seq) {
                None
            } else {
                Some(new_cell(store, run_mtp.then_some(mtp.as_ref()).flatten())?)
            });
        }
        let mut cells = Vec::with_capacity(rows.len());
        for (r, made) in rows.iter().zip(fresh) {
            let mut cell = match made.or_else(|| p.owners.take(r.seq)) {
                Some(c) => c,
                None => return Err(bad(format!("owner {} lost its cell", r.seq))),
            };
            if !run_mtp && cell.draft().is_some() {
                cell.drop_draft();
            }
            cell.hold_mark(r.ids.len());
            cells.push(cell);
        }
        let starts: Vec<usize> = cells.iter().map(OwnerCell::len).collect();
        let run = Run {
            model: &store.model,
            mtp: if run_mtp { mtp.as_ref() } else { None },
            kind,
            row_cap: p.row_cap,
            cancel,
        };
        let mut recorded = 0;
        let out = run.step(&mut p.adapter, rows, &mut cells, &starts, &mut recorded);
        let mut lost = Vec::new();
        for (i, (r, mut cell)) in rows.iter().zip(cells).enumerate() {
            if out.is_err() {
                let n = r.ids.len() as u32;
                let back = i >= recorded || p.adapter.rollback_last_tokens_for(r.seq, n).is_ok();
                if !back || !cell.rewind(starts[i], p.live_tokens(r.seq)) {
                    lost.push(r.seq);
                    continue;
                }
            } else if kind == RowKind::Prefill {
                cell.drop_mark();
            }
            p.owners.put(r.seq, cell)?;
        }
        for seq in lost {
            lose(p, seq, was == Some(seq));
        }
        if out.is_ok() && kind == RowKind::Decode && hold_turn {
            let mtp_loaded = mtp.is_some();
            for r in rows {
                hold_turn_state(p, r.seq, mtp_loaded);
            }
        }
        if let (Ok(_), Some(clock)) = (&out, clock) {
            clock.record(store, rows, waves);
        } else if out.is_ok() && kind == RowKind::Prefill {
            stats::emit_profile("prefill");
        }
        if let Some(seq) = was
            && p.owners.contains(seq)
        {
            p.owners
                .activate(seq, || Err(bad("the active cell vanished")))?;
            p.adapter.activate_request(seq).map_err(bad)?;
        }
        out
    }

    /// Bank and admission work due before the next step of a busy period:
    /// a period whose start was refused starts now or fails the step.
    fn safe_point(&mut self) -> Result<()> {
        let unleased = self
            .paged
            .as_ref()
            .is_some_and(|p| p.busy.is_live() && !self.in_turn);
        if unleased {
            self.begin()?;
            if let Some(p) = self.paged.as_mut() {
                p.safe.begun();
            }
        }
        if self.store.bank.is_direct() {
            return Ok(());
        }
        let Some(p) = self.paged.as_mut() else {
            return Ok(());
        };
        let due = p.safe.step();
        if let Err(e) = self.store.bank.safe_point(due.seed_reset) {
            tracing::warn!("hy_v4_flash safe point: {}", e.reason);
        }
        if !due.readmit {
            return Ok(());
        }
        let kv = p.pool_bytes() + p.state_bytes();
        let checked = self.store.recheck(kv);
        readmit(checked, || self.end())
    }

    /// Applies a busy-period edge: the lease at a begin and an end, a seed
    /// reset at the next safe point after a join.
    fn busy_edge(&mut self, edge: Edge) {
        match edge {
            Edge::Begin => {
                if let Err(e) = self.begin() {
                    tracing::warn!("hy_v4_flash busy period start: {}", e.reason);
                }
                if let Some(p) = self.paged.as_mut() {
                    p.safe.begun();
                }
            }
            Edge::Join => {
                if let Some(p) = self.paged.as_mut() {
                    p.safe.joined();
                }
            }
            Edge::End => self.end(),
            Edge::None => {}
        }
    }

    /// The paged mode's cancel flag edge (`set_turn_cancel_flag`).
    pub(crate) fn paged_flag(&mut self, set: bool) {
        let edge = match self.paged.as_mut() {
            Some(p) => p.busy.flag(set),
            None => return,
        };
        self.busy_edge(edge);
    }

    /// Makes `seq` the active owner (a new one gets an empty cell).
    fn activate_row(&mut self, seq: SeqId) -> Result<()> {
        let Inner {
            store, mtp, paged, ..
        } = self;
        let p = paged.as_mut().ok_or_else(no_pool)?;
        if p.owners.active() != Some(seq) || !p.owners.contains(seq) {
            p.owners.activate(seq, || new_cell(store, mtp.as_ref()))?;
            p.saved = p.live_tokens(seq).to_vec();
        }
        Ok(())
    }

    /// `seq`'s cell replaced by an empty one.
    fn fresh_row(&mut self, seq: SeqId) -> Result<()> {
        let Inner {
            store, mtp, paged, ..
        } = self;
        let p = paged.as_mut().ok_or_else(no_pool)?;
        p.owners.release(seq);
        p.owners.activate(seq, || new_cell(store, mtp.as_ref()))?;
        Ok(())
    }

    /// Prefill of `tokens` at the end of `seq`'s cell, in chunks ending at
    /// its turn's grid ends; the last chunk's last-row logits `[vocab]`
    /// with `logits`. A cancel fails with `PREFILL_CANCELLED`.
    fn prefill_rows(
        &mut self,
        seq: SeqId,
        tokens: &[u32],
        logits: bool,
    ) -> Result<Option<MxArray>> {
        let (base, ends, lookahead_prompt) = {
            let p = self.paged()?;
            let base = p.owners.get(seq).map_or(0, OwnerCell::len);
            let stop = base + tokens.len();
            let (mut ends, prompt): (Vec<usize>, Option<Vec<u32>>) = match p.turns.get(&seq) {
                Some(t) if t.prompt.len() >= stop && t.prompt[base..stop] == *tokens => (
                    self.grid
                        .ends(base, &t.prompt)
                        .into_iter()
                        .filter(|&e| e < stop)
                        .collect(),
                    Some(t.prompt[..stop].to_vec()),
                ),
                _ => (
                    (1..)
                        .map(|k| (base / self.grid.chunk() + k) * self.grid.chunk())
                        .take_while(|&e| e < stop)
                        .collect(),
                    None,
                ),
            };
            ends.push(stop);
            let prompt = if ends.len() >= 2 {
                Some(match prompt {
                    Some(pr) => pr,
                    None => {
                        let hist = p.live_tokens(seq);
                        let mut pr = Vec::with_capacity(base + tokens.len());
                        pr.extend_from_slice(&hist[..base.min(hist.len())]);
                        pr.extend_from_slice(tokens);
                        pr
                    }
                })
            } else {
                None
            };
            (base, ends, prompt)
        };
        let installed_lookahead = match lookahead_prompt.as_deref() {
            Some(prompt) if !self.store.model.has_prefill_engram(prompt, base) => {
                self.store.model.begin_prefill_engram(prompt, base, &ends);
                true
            }
            _ => false,
        };
        let flag = self.cancel.clone();
        let check = move |_: usize| cancelled(&flag);
        let layer_cancel: &dyn Fn(usize) -> bool = &check;
        let run_res = (|| -> Result<Option<MxArray>> {
            let mut out = None;
            let mut from = base;
            for end in ends {
                self.safe_point()?;
                if cancelled(&self.cancel) {
                    return Err(Error::from_reason(PREFILL_CANCELLED));
                }
                let last = end == base + tokens.len();
                let rows = [Rows {
                    seq,
                    ids: &tokens[from - base..end - base],
                    logits: if last && logits {
                        Logits::Last
                    } else {
                        Logits::None
                    },
                }];
                let outs = match self.wave_rows(
                    &rows,
                    RowKind::Prefill,
                    true,
                    false,
                    Some(layer_cancel),
                ) {
                    Ok(o) => o,
                    Err(_) if cancelled(&self.cancel) => {
                        return Err(Error::from_reason(PREFILL_CANCELLED));
                    }
                    Err(e) => return Err(e),
                };
                self.publish_at(seq, end);
                out = outs.into_iter().next().and_then(|o| o.logits);
                from = end;
            }
            out.map(|l| l.reshape(&[-1])).transpose()
        })();
        if installed_lookahead {
            self.store.model.clear_prefill_engram();
        }
        run_res
    }

    /// At a prompt state row of a publishing turn: registers `seq`'s blocks
    /// below `row` and publishes its cell there. A refusal costs reuse only.
    fn publish_at(&mut self, seq: SeqId, row: usize) {
        let mtp_loaded = self.mtp.is_some();
        let Some(p) = self.paged.as_mut() else {
            return;
        };
        let Some(plan) = p.turns.get(&seq) else {
            return;
        };
        let Some(kind) = plan.publish.then(|| plan.turn.prompt_kind(row)).flatten() else {
            return;
        };
        let blocks = row / BLOCK_ROWS;
        if plan.keys.len() < blocks || plan.prompt.len() < row {
            return;
        }
        let state = match p.owners.get_mut(seq).map(OwnerCell::state) {
            Some(Ok(Some(s))) if !mtp_loaded || s.draft.is_some() => s,
            Some(Err(e)) => {
                tracing::warn!("hy_v4_flash state at {row}: {}", e.reason);
                return;
            }
            _ => return,
        };
        if let Err(e) =
            Requests::register(&mut p.adapter, seq, row, &plan.keys[..blocks], plan.salt)
        {
            tracing::warn!("hy_v4_flash blocks below {row}: {}", e.reason);
            return;
        }
        let key = Key {
            tokens: &plan.prompt[..row],
            keys: &plan.keys[..blocks],
            salt: plan.salt,
        };
        match p.states.publish(key, state, kind) {
            Ok(Some(id)) => {
                p.published += 1;
                let replay = (kind != Kind::Checkpoint).then_some(id);
                disk_published(p, seq, row, id, replay);
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("hy_v4_flash state at {row}: {}", e.reason),
        }
    }

    /// One decode row per owner; logits `[rows, 1, vocab]`.
    fn decode_rows(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        let lanes = self.paged()?.lanes;
        if rows.len() > lanes {
            return Err(bad(format!("{} decode rows for {lanes} lanes", rows.len())));
        }
        self.safe_point()?;
        let ids: Vec<[u32; 1]> = rows.iter().map(|&(_, t)| [t]).collect();
        let wave: Vec<Rows<'_>> = rows
            .iter()
            .zip(&ids)
            .map(|(&(seq, _), id)| Rows {
                seq,
                ids: id,
                logits: Logits::Last,
            })
            .collect();
        let outs = self.wave_rows(&wave, RowKind::Decode, false, true, None)?;
        let mut logits = Vec::with_capacity(outs.len());
        for o in outs {
            let l = o
                .logits
                .ok_or_else(|| bad("a decode row returned no logits"))?;
            let v = l.shape_at(1)?;
            logits.push(l.reshape(&[1, 1, v])?);
        }
        match logits.len() {
            1 => Ok(logits.swap_remove(0)),
            _ => MxArray::concatenate_many(logits.iter().collect(), Some(0)),
        }
    }

    /// Admission of `tokens` on the active owner: the live cell continues,
    /// or a stored state resumes (RAM with a cached chain, else read back
    /// from disk, its blocks restored by the adapter's cold walk), or the
    /// prompt is cold. The turn's keys say how it writes its blocks.
    fn prime(&mut self, tokens: &[u32], reuse: bool, salt: u64) -> Result<Prefix> {
        let seq = self.paged()?.owners.active().unwrap_or(0);
        self.activate_paged_seq(seq)?;
        let (warm, cell_len) = {
            let p = self.paged_mut()?;
            p.turns.remove(&seq);
            let live = p.live_tokens(seq);
            let cell_len = p.owners.get(seq).map_or(0, OwnerCell::len);
            let warm = reuse
                && cell_len > 0
                && live.len() == cell_len
                && tokens.len() > cell_len
                && tokens.starts_with(live);
            (warm, cell_len)
        };
        if !warm {
            self.fresh_row(seq)?;
            let p = self.paged_mut()?;
            p.saved.clear();
            p.lines.remove(&seq);
        }
        let ends = self.grid.ends(0, tokens);
        let pass = pass_keys(tokens, &ends, self.grid.chunk());
        let nonce = rand::random::<u64>();
        let Inner {
            store,
            mtp,
            paged,
            grid,
            ..
        } = self;
        let p = paged.as_mut().ok_or_else(no_pool)?;
        let prompt = Key {
            tokens,
            keys: &pass,
            salt,
        };
        let hit = if reuse && !warm {
            let adapter = &p.adapter;
            p.states.prune(&|h| adapter.prefix_cached(h));
            p.states.lookup(prompt, &|h| adapter.prefix_cached(h))
        } else {
            None
        };
        let skip_disk = p.disk.as_mut().is_some_and(Disk::take_skip);
        let mut from_disk = None;
        let hit = if reuse && !warm && !skip_disk && p.disk.is_some() {
            match resume_from_disk(p, prompt, hit) {
                Some(deeper) => {
                    from_disk = Some(deeper.0);
                    Some(deeper)
                }
                None => hit,
            }
        } else {
            hit
        };
        let (from, line) = match hit {
            Some((id, row)) => (row, p.states.keys_of(id).unwrap_or_default()),
            None if warm => (cell_len, p.lines.get(&seq).cloned().unwrap_or_default()),
            None => (0, Vec::new()),
        };
        let keys = prompt_keys(tokens, from, &line, &pass, ends.contains(&from), nonce);
        let (skip_lookup, max_hit) = lookup_bounds(hit.map(|h| h.1), tokens.len());
        let plan = p
            .adapter
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq,
                tokens,
                tokens.len() as u32,
                warm,
                &keys,
                salt,
                skip_lookup,
                max_hit,
            )
            .map_err(bad)?;
        let expected = hit.map_or(if warm { cell_len } else { 0 }, |h| h.1);
        let cached = if plan.cached_prefix_len as usize != expected {
            if let Some(id) = from_disk {
                p.states.store_mut().drop_ram(id);
            }
            p.adapter.reset_for_new_request(seq).map_err(bad)?;
            p.adapter
                .allocate_suffix_blocks(tokens.len() as u32)
                .map_err(bad)?;
            p.owners.release(seq);
            p.owners.activate(seq, || new_cell(store, mtp.as_ref()))?;
            0
        } else if let Some((id, depth)) = hit {
            let state = p
                .states
                .get(id)
                .ok_or_else(|| bad("a stored state vanished at restore"))?;
            let draft = mtp.as_ref().map(|m| m.new_draft(store.ctx));
            let cell = OwnerCell::from_state(store.model.cfg(), state, &tokens[..depth], draft)?;
            p.owners.release(seq);
            p.owners.activate(seq, || Ok(cell))?;
            p.states.touch(id);
            p.restored += 1;
            if from_disk == Some(id)
                && let Some(disk) = p.disk.as_ref()
            {
                disk.restored();
            }
            depth
        } else {
            expected
        };
        let (keys, from) = if cached == from {
            (keys, from)
        } else {
            (prompt_keys(tokens, 0, &[], &pass, true, nonce), 0)
        };
        let turn = Turn::new(grid, cached, tokens);
        let last = turn.rows().last().map_or(0, |r| r.row);
        if reuse && last > from {
            disk_start(p, seq, tokens, &keys, salt, last);
        }
        p.turns.insert(
            seq,
            TurnPlan {
                prompt: tokens.to_vec(),
                keys,
                salt,
                publish: reuse,
                turn,
                nonce,
            },
        );
        Ok(Prefix {
            cached,
            suffix: tokens.len() - cached,
        })
    }

    /// The active owner's turn is over: with `reuse` its turn state
    /// publishes under write-path keys and the owner keeps the keys of its
    /// cell's blocks; its plan goes.
    fn close_turn(&mut self, reuse: bool) {
        let Some(p) = self.paged.as_mut() else {
            return;
        };
        let Some(seq) = p.owners.active() else {
            return;
        };
        let Some(mut plan) = p.turns.remove(&seq) else {
            return;
        };
        if reuse && plan.publish {
            publish_turn(p, seq, &mut plan);
        }
        if let Some(disk) = p.disk.as_mut() {
            disk.close(seq);
        }
        let Paged {
            disk,
            states,
            adapter,
            ..
        } = p;
        if let Some(disk) = disk.as_mut() {
            disk.step_now(states.store_mut(), &mut PoolBlocks(adapter));
        }
    }

    /// The active owner: an empty cell, no history, its request released.
    fn drop_active(&mut self) {
        let Some(seq) = self.paged.as_ref().and_then(|p| p.owners.active()) else {
            return;
        };
        if let Some(p) = self.paged.as_mut() {
            let _ = p.adapter.release_request_for(seq);
            p.turns.remove(&seq);
            p.lines.remove(&seq);
            p.saved.clear();
        }
        if let Err(e) = self.fresh_row(seq) {
            tracing::warn!("hy_v4_flash owner {seq}: {}", e.reason);
        }
    }

    /// `reset_caches` in paged mode: a command drops every owner, state and
    /// cached block; a prefix miss drops the active owner's history.
    pub(crate) fn paged_reset(&mut self, command: bool) -> Result<()> {
        if !command {
            let Some(p) = self.paged.as_mut() else {
                return Ok(());
            };
            p.saved.clear();
            if let Some(seq) = p.owners.active() {
                p.turns.remove(&seq);
                p.lines.remove(&seq);
                self.fresh_row(seq)?;
            }
            return Ok(());
        }
        let p = self.paged_mut()?;
        for seq in p.adapter.live_seq_ids() {
            p.adapter.release_request_for(seq).map_err(bad)?;
        }
        p.owners.clear();
        p.states.reset(p.disk.is_some());
        if let Some(disk) = p.disk.as_mut() {
            disk.reset();
        }
        p.turns.clear();
        p.lines.clear();
        p.saved.clear();
        p.adapter
            .release_request_and_purge_prefix_cache()
            .map_err(bad)?;
        self.waves.clear_open();
        Ok(())
    }

    /// `verify_cache_prefix` in paged mode: the saved history when the
    /// active owner still holds exactly it and the prompt extends it.
    pub(crate) fn paged_verify(&self, tokens: &[u32], reuse: bool) -> usize {
        let Some(p) = self.paged.as_ref() else {
            return 0;
        };
        let h = &p.saved;
        let live = p.owners.active().map_or(&[][..], |s| p.live_tokens(s));
        if reuse
            && !h.is_empty()
            && live == h.as_slice()
            && tokens.len() > h.len()
            && tokens.starts_with(h)
        {
            h.len()
        } else {
            0
        }
    }

    /// An exclusive turn's owner, refused when every lane holds another.
    fn activate_exclusive_seq(&mut self, seq: SeqId) -> Result<()> {
        let p = self.paged()?;
        if !p.owners.contains(seq) && p.owners.len() >= p.lanes {
            return Err(bad(
                "every lane holds an owner; releaseCacheOwner or resetCaches before adding another session",
            ));
        }
        self.activate_paged_seq(seq)
    }

    /// Mode-independent bytes of the sequences' caches.
    pub(crate) fn paged_kv_bytes(&self) -> Option<u64> {
        self.paged
            .as_ref()
            .map(|p| p.pool_bytes() + p.state_bytes())
    }

    /// Bytes outside the pool the admission charged for: every lane's
    /// floor and draft keys, the RAM states, and the MTP rows of the
    /// admitted context.
    fn state_budget(&self) -> u64 {
        let cfg = self.store.model.cfg();
        let mtp = self.mtp.is_some();
        let lanes = self.paged.as_ref().map_or(1, |p| p.lanes);
        let mtp_row = kv_per_token(cfg, mtp).saturating_sub(kv_per_token(cfg, false));
        (lanes as u64) * lane_bytes(cfg, mtp, self.store.ctx)
            + ladder_bytes_for(lanes)
            + mtp_row * self.store.ctx as u64
    }

    /// MTP rows `tokens` adds to a sequence.
    fn draft_bytes(&self, tokens: u64) -> u64 {
        let cfg = self.store.model.cfg();
        let mtp = self.mtp.is_some();
        kv_per_token(cfg, mtp).saturating_sub(kv_per_token(cfg, false)) * tokens
    }
}

impl Inner {
    /// Positions `seq`'s cell holds (0 without one).
    pub(super) fn owner_len(&self, seq: SeqId) -> usize {
        self.paged
            .as_ref()
            .and_then(|p| p.owners.get(seq))
            .map_or(0, OwnerCell::len)
    }

    /// `seq`'s draft cut to the rows below its last trunk row: the drafts
    /// of a proposal go, the exact rows stay.
    pub(super) fn settle(&mut self, seq: SeqId) -> Result<()> {
        let p = self.paged_mut()?;
        let Some(cell) = p.owners.get_mut(seq) else {
            return Ok(());
        };
        let keep = cell.len().saturating_sub(1);
        match cell.draft_mut() {
            Some(d) if d.len() > keep => d.cut(keep),
            _ => Ok(()),
        }
    }

    /// Whether `seq` stands settled at `position`: a cell of that length
    /// whose draft holds the rows below it and the seed of its last row.
    fn settled_at(&self, seq: SeqId, position: usize) -> bool {
        let Some(cell) = self.paged.as_ref().and_then(|p| p.owners.get(seq)) else {
            return position == 0;
        };
        let Some(d) = cell.draft() else {
            return false;
        };
        cell.len() == position
            && match position.checked_sub(1) {
                Some(last) => d.len() == last && d.seed(last).is_some(),
                None => d.is_empty(),
            }
    }

    /// One draft row at the end of `seq`'s draft from `h` and `emb`, deferring
    /// KV/indexer eval to the caller's scope.
    pub(super) fn draft_row_lazy_with_persist(
        &mut self,
        seq: SeqId,
        h: &MxArray,
        emb: &MxArray,
        persist_state: bool,
    ) -> Result<(MxArray, MxArray)> {
        let Inner {
            store, mtp, paged, ..
        } = self;
        let m = mtp.as_ref().ok_or_else(no_mtp)?;
        let p = paged.as_mut().ok_or_else(no_pool)?;
        let d = p
            .owners
            .get_mut(seq)
            .and_then(OwnerCell::draft_mut)
            .ok_or_else(|| bad(format!("owner {seq} has no draft")))?;
        m.draft_lazy_with_persist(&store.model, d, h, emb, persist_state)
    }

    /// `n` drafts after `anchor` on `seq`'s settled draft, drawn as the
    /// target samples; the draft is settled again after.
    fn propose(
        &mut self,
        seq: SeqId,
        anchor: u32,
        n: usize,
        params: &ChatParams,
        rng: &mut dyn Rng,
    ) -> Result<DsparkProposal> {
        if n == 0 {
            return Ok(empty_proposal());
        }
        self.settle(seq)?;
        let last = self
            .owner_len(seq)
            .checked_sub(1)
            .ok_or_else(|| bad(format!("owner {seq} proposes on an empty cell")))?;
        let seed = self
            .paged()?
            .owners
            .get(seq)
            .and_then(OwnerCell::draft)
            .and_then(|d| d.seed(last))
            .cloned()
            .ok_or_else(|| bad(format!("owner {seq} has no seed for row {last}")))?;
        let drawn = self.draw(seq, seed, anchor, n, params, rng);
        let settled = self.settle(seq);
        let out = drawn?;
        settled.map(|()| out)
    }

    fn draw(
        &mut self,
        seq: SeqId,
        mut h: MxArray,
        anchor: u32,
        n: usize,
        params: &ChatParams,
        rng: &mut dyn Rng,
    ) -> Result<DsparkProposal> {
        let _scope = stats::DeferScope::enter();
        let mut out = empty_proposal();
        let target = params.sampling_config.unwrap_or_default();
        let greedy = crate::sampling::is_greedy_temperature(target.temperature.unwrap_or(1.0));
        let cfg = crate::engine::mtp_decode::mtp_draft_sampling_config(target);
        let mut tok = MxArray::from_uint32(&[anchor], &[1])?;
        let mut step_ids: Vec<MxArray> = Vec::with_capacity(n);
        let mut step_dists: Vec<MxArray> = Vec::new();
        for step in 0..n {
            let emb = self.store.model.weights().token_embd().rows(&tok)?;
            let (next, logits) = self.draft_row_lazy_with_persist(seq, &h, &emb, step + 1 < n)?;
            let logits = logits.reshape(&[-1])?;
            let (id, dist) = if greedy {
                (logits.argmax(-1, None)?.astype(DType::Int32)?, None)
            } else {
                let dist = crate::sampling::sampling_distribution(&logits, Some(cfg))?
                    .astype(DType::Float32)?;
                let id = crate::sampling::sample_dense_distribution_array(&dist, &mut *rng)?;
                (id, Some(dist))
            };
            let id_1d = id.reshape(&[1])?;
            if let Some(d) = dist {
                step_dists.push(d);
            }
            if step + 1 < n {
                tok = id_1d.clone();
                h = next;
            }
            step_ids.push(id_1d);
        }
        let mut eval_targets: Vec<&MxArray> = Vec::with_capacity(step_ids.len() + step_dists.len());
        eval_targets.extend(step_ids.iter());
        eval_targets.extend(step_dists.iter());
        stats::eval(&eval_targets, "hy_v4_flash MTP proposal")?;
        for id in &step_ids {
            let v = id.item_at_int32(0)?;
            u32::try_from(v).map_err(|_| bad(format!("draft id {v}")))?;
            out.draft_ids.push(v);
        }
        out.draft_dists = step_dists;
        Ok(out)
    }

    /// Batched MTP proposals across all eligible lanes: shares `embed_tokens`,
    /// `eh_proj`, `mla.project_at`, `mla.finish_with_gate`, `ffn`,
    /// `shared_head_norm`, and the vocab head in one pass per depth step, with
    /// lazy GPU token-to-embedding chaining and a single `stats::eval` across
    /// all depth steps.
    fn propose_batch(
        &mut self,
        requests: &mut [ScheduledProposalRequest<'_>],
    ) -> Vec<Result<DsparkProposal>> {
        if requests.is_empty() {
            return Vec::new();
        }
        if self.mtp.is_none() {
            return requests.iter().map(|_| Err(no_mtp())).collect();
        }
        let grants: Vec<usize> = match self.paged_mut() {
            Ok(p) => {
                let row_cap = p.row_cap;
                requests
                    .iter()
                    .map(|req| p.spec.budget.grant(req.seq_id, req.max_drafts, row_cap))
                    .collect()
            }
            Err(e) => {
                return requests
                    .iter()
                    .map(|_| Err(Error::from_reason(e.reason.clone())))
                    .collect();
            }
        };

        struct ActiveLane {
            req_idx: usize,
            seq: SeqId,
            grant: usize,
            greedy: bool,
            cfg: crate::sampling::SamplingConfig,
            h: MxArray,
            tok: MxArray,
            step_ids: Vec<MxArray>,
            step_dists: Vec<MxArray>,
        }

        let mut results: Vec<Result<DsparkProposal>> =
            (0..requests.len()).map(|_| Ok(empty_proposal())).collect();
        let mut active: Vec<ActiveLane> = Vec::with_capacity(requests.len());
        let mut seen = HashSet::with_capacity(requests.len());

        for (i, (&grant, req)) in grants.iter().zip(requests.iter()).enumerate() {
            if grant == 0 {
                continue;
            }
            if !seen.insert(req.seq_id) {
                results[i] = Err(bad(format!(
                    "duplicate owner {} in proposal batch",
                    req.seq_id
                )));
                continue;
            }
            if let Err(e) = self.settle(req.seq_id) {
                results[i] = Err(e);
                continue;
            }
            let last = match self.owner_len(req.seq_id).checked_sub(1) {
                Some(l) => l,
                None => {
                    results[i] = Err(bad(format!(
                        "owner {} proposes on an empty cell",
                        req.seq_id
                    )));
                    continue;
                }
            };
            let seed = match self
                .paged()
                .ok()
                .and_then(|p| p.owners.get(req.seq_id))
                .and_then(OwnerCell::draft)
                .and_then(|d| d.seed(last))
                .cloned()
            {
                Some(s) => s,
                None => {
                    results[i] = Err(bad(format!(
                        "owner {} has no seed for row {last}",
                        req.seq_id
                    )));
                    continue;
                }
            };
            let tok = match MxArray::from_uint32(&[req.anchor], &[1]) {
                Ok(t) => t,
                Err(e) => {
                    results[i] = Err(e);
                    continue;
                }
            };
            let target = req.params.sampling_config.unwrap_or_default();
            let greedy = crate::sampling::is_greedy_temperature(target.temperature.unwrap_or(1.0));
            let cfg = crate::engine::mtp_decode::mtp_draft_sampling_config(target);
            active.push(ActiveLane {
                req_idx: i,
                seq: req.seq_id,
                grant,
                greedy,
                cfg,
                h: seed,
                tok,
                step_ids: Vec::with_capacity(grant),
                step_dists: Vec::new(),
            });
        }

        if active.is_empty() {
            return results;
        }

        let max_depth = active.iter().map(|l| l.grant).max().unwrap_or(0);
        let dim = self.store.model.cfg().dim as i64;
        let (was, mut cells) = match self.paged_mut() {
            Ok(p) => {
                let was = p.owners.active();
                let mut cells = Vec::with_capacity(active.len());
                let mut taken_ok = true;
                for lane in &active {
                    match p.owners.take(lane.seq) {
                        Some(cell) => cells.push((lane.seq, cell)),
                        None => {
                            taken_ok = false;
                            break;
                        }
                    }
                }
                if !taken_ok {
                    for (seq, cell) in cells {
                        let _ = p.owners.put(seq, cell);
                    }
                    let _ = reactivate(p, was);
                    for lane in &active {
                        results[lane.req_idx] =
                            Err(bad(format!("owner {} lost its cell", lane.seq)));
                    }
                    return results;
                }
                (was, cells)
            }
            Err(e) => {
                for lane in &active {
                    results[lane.req_idx] = Err(Error::from_reason(e.reason.clone()));
                }
                return results;
            }
        };

        let run_res: Result<()> = (|| {
            let tp_mtp = stats::prof_start();
            let _scope = stats::DeferScope::enter();
            let m = self.mtp.as_ref().ok_or_else(no_mtp)?;
            let model = &self.store.model;
            let token_embd = model.weights().token_embd();

            for step in 0..max_depth {
                let step_indices: Vec<usize> = active
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, lane)| (step < lane.grant).then_some(idx))
                    .collect();
                let b = step_indices.len();
                if b == 0 {
                    break;
                }

                let tok_batch = if b == 1 {
                    active[step_indices[0]].tok.clone()
                } else {
                    let toks: Vec<&MxArray> =
                        step_indices.iter().map(|&idx| &active[idx].tok).collect();
                    MxArray::concatenate_many(toks, Some(0))?
                };
                let emb_batch = token_embd.rows(&tok_batch)?;

                let h_rows: Vec<MxArray> = step_indices
                    .iter()
                    .map(|&idx| active[idx].h.reshape(&[1, dim]))
                    .collect::<Result<_>>()?;
                let h_batch = if b == 1 {
                    h_rows[0].clone()
                } else {
                    MxArray::concatenate_many(h_rows.iter().collect(), Some(0))?
                };

                let mut step_set = vec![false; cells.len()];
                for &idx in &step_indices {
                    step_set[idx] = true;
                }
                let mut drafts: Vec<&mut super::mtp::Draft> = Vec::with_capacity(b);
                for (idx, (seq, cell)) in cells.iter_mut().enumerate() {
                    if step_set[idx] {
                        let d = cell
                            .draft_mut()
                            .ok_or_else(|| bad(format!("owner {seq} has no draft")))?;
                        drafts.push(d);
                    }
                }

                let persist_lanes: Vec<bool> = step_indices
                    .iter()
                    .map(|&idx| step + 1 < active[idx].grant)
                    .collect();
                let (h_next, logits) = m.draft_batch_with_persist(
                    model,
                    &mut drafts,
                    &h_batch,
                    &emb_batch,
                    &persist_lanes,
                )?;
                let all_greedy = step_indices.iter().all(|&idx| active[idx].greedy);
                let vocab = logits.shape_at(1)?;

                if all_greedy {
                    let argmax_all = logits.argmax(-1, None)?.astype(DType::Int32)?;
                    for (j, &idx) in step_indices.iter().enumerate() {
                        let lo = j as i64;
                        let hi = lo + 1;
                        let id_1d = if b == 1 {
                            argmax_all.reshape(&[1])?
                        } else {
                            argmax_all.slice(&[lo], &[hi])?
                        };
                        let lane = &mut active[idx];
                        if step + 1 < lane.grant {
                            lane.tok = id_1d.clone();
                            lane.h = if b == 1 {
                                h_next.clone()
                            } else {
                                h_next.slice(&[lo, 0], &[hi, dim])?
                            };
                        }
                        lane.step_ids.push(id_1d);
                    }
                } else {
                    for (j, &idx) in step_indices.iter().enumerate() {
                        let lo = j as i64;
                        let hi = lo + 1;
                        let row_logits = if b == 1 {
                            logits.reshape(&[-1])?
                        } else {
                            logits.slice(&[lo, 0], &[hi, vocab])?.reshape(&[-1])?
                        };
                        let lane = &mut active[idx];
                        let (id_1d, dist) = if lane.greedy {
                            (
                                row_logits
                                    .argmax(-1, None)?
                                    .astype(DType::Int32)?
                                    .reshape(&[1])?,
                                None,
                            )
                        } else {
                            let dist = crate::sampling::sampling_distribution(
                                &row_logits,
                                Some(lane.cfg),
                            )?
                            .astype(DType::Float32)?;
                            let id = crate::sampling::sample_dense_distribution_array(
                                &dist,
                                &mut *requests[lane.req_idx].rng,
                            )?
                            .reshape(&[1])?;
                            (id, Some(dist))
                        };
                        if let Some(d) = dist {
                            lane.step_dists.push(d);
                        }
                        if step + 1 < lane.grant {
                            lane.tok = id_1d.clone();
                            lane.h = if b == 1 {
                                h_next.clone()
                            } else {
                                h_next.slice(&[lo, 0], &[hi, dim])?
                            };
                        }
                        lane.step_ids.push(id_1d);
                    }
                }
            }

            let mut eval_targets: Vec<&MxArray> = Vec::new();
            for lane in &active {
                eval_targets.extend(lane.step_ids.iter());
                eval_targets.extend(lane.step_dists.iter());
            }
            stats::eval(&eval_targets, "hy_v4_flash MTP proposal batch")?;
            stats::prof_end(stats::Phase::Mtp, tp_mtp, &[])?;
            Ok(())
        })();

        if let Ok(p) = self.paged_mut() {
            for (seq, cell) in cells {
                let _ = p.owners.put(seq, cell);
            }
            let _ = reactivate(p, was);
        }

        for lane in &active {
            let settle_res = self.settle(lane.seq);
            if let Err(e) = &run_res {
                results[lane.req_idx] = Err(Error::from_reason(e.reason.clone()));
                continue;
            }
            if let Err(e) = settle_res {
                results[lane.req_idx] = Err(e);
                continue;
            }
            let mut proposal = empty_proposal();
            let mut lane_err = None;
            for id in &lane.step_ids {
                match id.item_at_int32(0) {
                    Ok(v) if u32::try_from(v).is_ok() => proposal.draft_ids.push(v),
                    Ok(v) => {
                        lane_err = Some(bad(format!("draft id {v}")));
                        break;
                    }
                    Err(e) => {
                        lane_err = Some(e);
                        break;
                    }
                }
            }
            if let Some(e) = lane_err {
                results[lane.req_idx] = Err(e);
            } else {
                proposal.draft_dists = lane.step_dists.clone();
                results[lane.req_idx] = Ok(proposal);
            }
        }
        results
    }

    /// One decode wave of `rows` (distinct owners) on settled drafts: every
    /// owner's trunk rows and the MTP rows they close. `hold_turn` is true for
    /// committed steps (`spec::Paged::forward_with_hidden`) and false for
    /// speculative verify waves (`verify_scheduled`, `spec::Paged::verify_step`).
    pub(super) fn mtp_wave(
        &mut self,
        rows: &[(SeqId, &[u32])],
        logits: Logits,
        hold_turn: bool,
    ) -> Result<Vec<Output>> {
        let lanes = self.paged()?.lanes;
        if rows.len() > lanes {
            return Err(bad(format!(
                "{} owners in a wave for {lanes} lanes",
                rows.len()
            )));
        }
        self.safe_point()?;
        for &(seq, _) in rows {
            self.settle(seq)?;
        }
        let wave: Vec<Rows<'_>> = rows
            .iter()
            .map(|&(seq, ids)| Rows { seq, ids, logits })
            .collect();
        self.wave_rows(&wave, RowKind::Decode, true, hold_turn, None)
    }

    /// `seq`'s pool rows and cell cut to `len`; an owner that cannot go
    /// there is dropped.
    pub(super) fn cut_owner(&mut self, seq: SeqId, len: usize) -> Result<()> {
        let mtp_loaded = self.mtp.is_some();
        let cut = self.paged_mut()?.cut_owner(seq, len, mtp_loaded)?;
        if cut > 0 {
            self.waves.cut(seq, cut);
        }
        Ok(())
    }

    /// Reserves `rows` more pool rows for `seq`; false when they do not
    /// fit (`rows_fit`) or the pool refuses.
    pub(super) fn reserve_rows(&mut self, seq: SeqId, rows: usize) -> Result<bool> {
        self.activate_paged_seq(seq)?;
        let len = self.owner_len(seq);
        let ctx = self.store.ctx;
        let p = self.paged_mut()?;
        if !rows_fit(rows, len, ctx, p.row_cap) {
            return Ok(false);
        }
        let rows = u32::try_from(rows).map_err(|_| bad("rows exceed u32"))?;
        Ok(p.adapter.reserve_rows(rows).is_ok())
    }

    /// The scheduled verify: every row in one wave, logits
    /// `[rows, 1, vocab]` in row order; each owner's rows stay open until
    /// its commit.
    fn verify_scheduled(&mut self, rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
        let p = self.paged_mut()?;
        p.spec.budget.close();
        if !p.spec.open.is_empty() {
            return Err(bad("a verify while another is open"));
        }
        let mut spans = Vec::with_capacity(rows.len());
        for r in rows {
            let start = self.owner_len(r.seq_id);
            if r.tokens.is_empty()
                || r.tokens.len() > LOOKAHEAD_MAX
                || r.first_position as usize != start
            {
                return Err(bad(format!(
                    "owner {}: {} verify rows at {} on a cell of {start}",
                    r.seq_id,
                    r.tokens.len(),
                    r.first_position
                )));
            }
            spans.push((r.seq_id, r.tokens.as_slice()));
        }
        let outs = self.mtp_wave(&spans, Logits::All, false)?;
        let joined = join_logits(outs, rows);
        let p = self.paged_mut()?;
        for r in rows {
            p.spec.open.insert(
                r.seq_id,
                Open {
                    start: r.first_position as usize,
                    rows: r.tokens.len(),
                },
            );
        }
        if joined.is_err() {
            for r in rows {
                let _ = self.close_verify(r.seq_id, 0);
            }
        }
        joined
    }

    /// Closes `seq`'s open verify keeping `keep` rows.
    fn close_verify(&mut self, seq: SeqId, keep: usize) -> Result<()> {
        let open = self
            .paged_mut()?
            .spec
            .open
            .remove(&seq)
            .ok_or_else(|| bad(format!("owner {seq} commits with no open verify")))?;
        let len = match commit_len(open, keep) {
            Ok(len) => len,
            Err(e) => {
                let _ = self.cut_owner(seq, open.start);
                return Err(e);
            }
        };
        self.cut_owner(seq, len)
    }

    fn commit_scheduled(&mut self, rows: &[ScheduledVerifyCommit]) -> Vec<Result<()>> {
        let out = rows
            .iter()
            .map(|c| self.close_verify(c.seq_id, c.keep))
            .collect();
        let left: Vec<SeqId> = self
            .paged
            .as_ref()
            .map(|p| p.spec.open.keys().copied().collect())
            .unwrap_or_default();
        for seq in left {
            let _ = self.close_verify(seq, 0);
        }
        out
    }
}

/// The verify's per-owner logits `[n, vocab]` as `[rows, 1, vocab]`.
fn join_logits(outs: Vec<Output>, rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
    let mut parts = Vec::with_capacity(outs.len());
    for (o, r) in outs.into_iter().zip(rows) {
        let l = o
            .logits
            .ok_or_else(|| bad("a verify row returned no logits"))?;
        if l.shape_at(0)? as usize != r.tokens.len() {
            return Err(bad(format!(
                "owner {}: {} logits rows for {} verify rows",
                r.seq_id,
                l.shape_at(0)?,
                r.tokens.len()
            )));
        }
        parts.push(l);
    }
    let all = match parts.len() {
        0 => return Err(bad("a verify of no rows")),
        1 => parts.swap_remove(0),
        _ => MxArray::concatenate_many(parts.iter().collect(), Some(0))?,
    };
    all.expand_dims(1)
}

pub struct Prefix {
    pub cached: usize,
    pub suffix: usize,
}

impl PagedPrefix for Prefix {
    fn effective_cached_prefix_len(&self) -> usize {
        self.cached
    }

    fn suffix_len(&self) -> usize {
        self.suffix
    }
}

/// The exclusive lane's decode step of the active owner.
pub(crate) struct PagedStep<'a>(&'a mut Inner);

impl PagedStepModel for PagedStep<'_> {
    const EVAL: EvalPolicy = EvalPolicy::AsyncTokenAndForcedLogits;
    /// The cells hold only forwarded rows: a replayed step would desync the
    /// history `save_paged_history` publishes.
    const FINAL_TOKEN_POLICY: FinalTokenPolicy = FinalTokenPolicy::AlwaysDrop;

    fn paged_step(&mut self, token: u32) -> Result<MxArray> {
        let seq = self
            .0
            .paged()?
            .owners
            .active()
            .ok_or_else(|| bad("a decode step without an active owner"))?;
        self.0.decode_rows(&[(seq, token)])
    }
}

impl PagedBackend for Inner {
    type PagedDecode<'a> = PagedStepper<PagedStep<'a>>;
    type PrefixState = Prefix;

    fn prime_prefix_state(
        &mut self,
        tokens: &[u32],
        reuse: bool,
        _: usize,
        _: &[u64],
        salt: u64,
    ) -> Result<Prefix> {
        self.prime(tokens, reuse, salt)
    }

    fn paged_prefill(&mut self, tokens: &[u32], _: &Prefix, stream: Stream) -> Result<MxArray> {
        let _ctx = StreamContext::new(stream);
        let seq = self
            .paged()?
            .owners
            .active()
            .ok_or_else(|| bad("a prefill without an active owner"))?;
        self.prefill_rows(seq, tokens, true)?
            .ok_or_else(|| bad("the prefill returned no logits"))
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(PagedStepper(PagedStep(self)))
    }

    /// The owner stays live without publishing its blocks: rows a decode
    /// wrote are not what a cold prefill writes. States published during
    /// the prefill stay.
    fn finalize_paged_turn(&mut self, reuse: bool, _: u64) {
        self.close_turn(reuse);
        let Some(p) = self.paged.as_mut() else {
            return;
        };
        let kept = if reuse {
            p.adapter.finalize_turn_keep_live_no_prefix()
        } else {
            p.adapter.release_request().map(|_| ())
        };
        if kept.is_err() {
            self.drop_active();
        }
    }

    fn save_paged_history(&mut self, _: &[u32], _: &[u32], _: bool, reuse: bool) -> Result<()> {
        if reuse {
            let p = self.paged_mut()?;
            p.saved = p.adapter.request_tokens().to_vec();
        } else {
            self.drop_active();
        }
        Ok(())
    }

    fn abort_paged_turn(&mut self) {
        self.drop_active();
    }

    /// An exclusive MTP turn on the active owner, its first cycle's rows
    /// reserved; false (AR) when they do not fit.
    fn admit_paged_speculative_decode(&mut self, params: &ChatParams) -> Result<bool> {
        if self.mtp.is_none() {
            return Ok(false);
        }
        let seq = self
            .paged()?
            .owners
            .active()
            .ok_or_else(|| bad("an MTP turn without an active owner"))?;
        self.settle(seq)?;
        if !self.settled_at(seq, self.owner_len(seq)) {
            return Ok(false);
        }
        self.reserve_rows(seq, turn_lookahead_rows(spec::PAGED_PLAN, params))
    }

    fn run_paged_speculative_decode(
        &mut self,
        args: PagedSpeculativeArgs<'_, '_, '_>,
    ) -> Result<()> {
        let PagedSpeculativeArgs {
            y,
            params,
            reasoning_tracker,
            profiler,
            max_new_tokens,
            eos_id,
            generated_tokens,
            token_history,
            finish_reason,
            first_token_instant,
            report_perf,
            generation_stream,
            cancel_flag,
            streaming,
            ..
        } = args;
        MxArray::async_eval_arrays(&[&y]);
        let mut rng = rand::rng();
        run_mtp_turn(
            self,
            &mut rng,
            MtpTurnArgs {
                y,
                depth: params.mtp_depth,
                params,
                reasoning_tracker,
                profiler,
                max_new_tokens,
                eos_id,
                generated_tokens,
                token_history,
                finish_reason,
                first_token_instant,
                report_perf,
                generation_stream,
                prompt_hidden: None,
                prompt_hidden_ids: None,
                cancel_flag,
            },
            streaming,
        )
        .map(drop)
    }
}

impl HybridSchedulerBackend for Inner {
    type FamilyCommand = std::convert::Infallible;
    type RestoreTicket = NoRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;
    const SCHEDULER_NAME: &'static str = "hy_v4_flash";

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged.as_ref().map(|p| &p.adapter)
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged.as_mut().map(|p| &mut p.adapter)
    }

    fn ownerless_session_in_owner_tables(&self) -> bool {
        true
    }

    fn supports_scheduled_speculation(&self) -> bool {
        self.paged.is_some() && self.mtp.is_some()
    }

    /// The owner speculates from its prefill when its draft stands settled
    /// at `position` (a new cell at row 0, a live or resumed one at its end).
    fn begin_scheduled_speculation(&mut self, seq: SeqId, position: u32) -> Result<bool> {
        if !self.supports_scheduled_speculation() {
            return Ok(false);
        }
        self.activate_row(seq)?;
        self.settle(seq)?;
        Ok(self.settled_at(seq, position as usize))
    }

    fn reserve_scheduled_speculation(&mut self, seq: SeqId, queries: usize) -> Result<bool> {
        let rows = self.paged()?.spec.budget.rows_for(seq, queries);
        let fits = self.reserve_rows(seq, rows)?;
        if fits {
            self.paged_mut()?.spec.budget.reserved_rows(seq, rows);
        }
        Ok(fits)
    }

    fn propose_scheduled(
        &mut self,
        seq: SeqId,
        anchor: u32,
        cap: usize,
        params: &ChatParams,
        rng: &mut dyn Rng,
        _confidence: bool,
    ) -> Result<DsparkProposal> {
        let p = self.paged_mut()?;
        let row_cap = p.row_cap;
        let n = p.spec.budget.grant(seq, cap, row_cap);
        self.propose(seq, anchor, n, params, rng)
    }

    fn propose_scheduled_batch(
        &mut self,
        requests: &mut [ScheduledProposalRequest<'_>],
    ) -> Vec<Result<DsparkProposal>> {
        self.propose_batch(requests)
    }

    fn run_scheduled_verify(&mut self, rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
        self.verify_scheduled(rows)
    }

    fn commit_scheduled_verify(
        &mut self,
        rows: &[ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        Ok(self.commit_scheduled(rows))
    }

    /// An open verify of `seq` goes back to its start.
    fn release_scheduled_speculation(&mut self, seq: SeqId) {
        let open = self
            .paged
            .as_ref()
            .is_some_and(|p| p.spec.open.contains_key(&seq));
        if open && let Err(e) = self.close_verify(seq, 0) {
            tracing::warn!("hy_v4_flash owner {seq}: {}", e.reason);
        }
    }

    fn scheduler_capacity(&self) -> usize {
        self.paged.as_ref().map_or(1, |p| p.lanes)
    }

    fn scheduled_turn_opened(&mut self, seq: SeqId) {
        let edge = match self.paged.as_mut() {
            Some(p) => p.busy.opened(seq),
            None => return,
        };
        self.busy_edge(edge);
    }

    /// The turn's plan stays for its finalize, which publishes its turn
    /// state, or its abort or release, which drop it.
    fn scheduled_turn_closed(&mut self, seq: SeqId) {
        let edge = match self.paged.as_mut() {
            Some(p) => p.busy.closed(seq),
            None => return,
        };
        self.busy_edge(edge);
    }

    fn release_scheduled_cache(&mut self, seq: SeqId) -> Result<()> {
        let Some(p) = self.paged.as_mut() else {
            return Ok(());
        };
        p.turns.remove(&seq);
        if let Some(disk) = p.disk.as_mut() {
            disk.close(seq);
        }
        if p.adapter.block_table_for(seq).is_some() {
            p.adapter.release_request_for(seq).map_err(bad)?;
        }
        self.waves.forget(seq);
        Ok(())
    }

    fn tick_state_disk(&mut self, _: crate::models::qwen3_5::dflash2_state::disk::Tick) -> bool {
        self.paged.as_mut().is_some_and(Paged::top_up_disk)
    }

    fn scheduler_prefill_slice_tokens(&self) -> u32 {
        u32::try_from(self.grid.chunk()).unwrap_or(u32::MAX)
    }

    /// A prefill slice is one chunk a cold prefill of the prompt runs (C
    /// rows on its grid), outside the engine's step row budget, whatever
    /// rows the decoding lanes take.
    fn scheduler_whole_prompt_prefill(&self, _: u32, _: u32) -> bool {
        self.paged.is_some()
    }

    /// The slice ends of the active owner's turn: the engine reads them
    /// right after `prime_prefix_state` ran on that owner.
    fn extra_prefill_breaks(&self, prompt_tokens: u32, cached_prefix: u32) -> Vec<u32> {
        let prompt = self.paged.as_ref().and_then(|p| {
            p.owners
                .active()
                .and_then(|seq| p.turns.get(&seq))
                .map(|t| t.prompt.as_slice())
        });
        slice_breaks(&self.grid, prompt, prompt_tokens, cached_prefix)
    }

    /// A step's rows open the next verify wave.
    fn claim_step_rows(state: &mut HybridSchedulerState<Self>, plan: &StepPlan) -> StepClaims {
        if let Some(p) = state.inner.paged.as_mut() {
            p.spec.budget.close();
            let decode: Vec<(SeqId, usize)> = plan
                .rows
                .iter()
                .filter(|r| r.kind == StepKind::Decode)
                .map(|r| (r.seq_id, r.num_tokens as usize))
                .collect();
            let row_cap = p.row_cap;
            p.spec.budget.plan(&decode, row_cap);
        }
        StepClaims::granted(plan)
    }

    fn scheduler_books_on_demand(&self) -> bool {
        true
    }

    fn planned_turn_window(&self, _speculative: bool) -> Option<u32> {
        self.paged
            .as_ref()
            .and_then(|_| u32::try_from(self.store.ctx).ok())
    }

    fn max_position_embeddings(&self) -> i32 {
        i32::try_from(self.store.ctx).unwrap_or(i32::MAX)
    }

    fn memory_accounts(&self) -> Option<MemoryAccounts> {
        let p = self.paged.as_ref()?;
        Some(MemoryAccounts {
            kv: p.pool_bytes(),
            states: p.state_bytes(),
            embeddings: 0,
        })
    }

    fn memory_model(&self) -> u64 {
        self.paged.as_ref().map_or(0, |p| p.governor.model_id())
    }

    fn memory_governor(&self) -> Option<&'static Governor> {
        self.paged.as_ref().map(|p| p.governor.governor())
    }

    /// Owner cells and RAM states: a load's warm-up must leave none.
    fn retained_state_entries(&self) -> usize {
        self.paged
            .as_ref()
            .map_or(0, |p| p.owners.len() + p.states.ram_len())
    }

    fn scheduled_state_capacity(&self) -> Option<ScheduledStateCapacity> {
        let p = self.paged.as_ref()?;
        Some(ScheduledStateCapacity {
            budget: self.state_budget(),
            held: p.state_bytes(),
        })
    }

    /// RAM states toward the shortfall, in eviction order: owner cells
    /// outrank them.
    fn reclaim_for_scheduled_state(&mut self, shortfall: u64) -> bool {
        self.paged
            .as_mut()
            .is_some_and(|p| p.states.reclaim(shortfall) > 0)
    }

    fn scheduled_state_candidate_bytes(&self, seq: SeqId, tokens: u32, _: bool) -> u64 {
        let Some(p) = self.paged.as_ref() else {
            return 0;
        };
        let cfg = self.store.model.cfg();
        let held = p.owners.get(seq).map_or(0, OwnerCell::len) as u64;
        let floor = if p.owners.contains(seq) {
            0
        } else {
            lane_bytes(cfg, self.mtp.is_some(), self.store.ctx)
        };
        floor + self.draft_bytes(u64::from(tokens).saturating_sub(held))
    }

    fn scheduled_state_held_bytes(&self, seq: SeqId) -> u64 {
        self.paged.as_ref().map_or(0, |p| p.owners.held_bytes(seq))
    }

    fn has_scheduled_recurrent(&self, seq: SeqId) -> bool {
        self.paged.as_ref().is_some_and(|p| p.owners.contains(seq))
    }

    fn activate_scheduled_recurrent(&mut self, seq: SeqId) -> Result<()> {
        self.activate_row(seq)
    }

    fn activate_paged_seq(&mut self, seq: SeqId) -> Result<()> {
        self.activate_row(seq)?;
        let p = self.paged_mut()?;
        if p.adapter.block_table_for(seq).is_some() {
            p.adapter.activate_request(seq)
        } else {
            p.adapter.begin_request(seq)
        }
        .map_err(bad)
    }

    fn park_active_scheduled_recurrent(&mut self) -> Result<()> {
        if let Some(p) = self.paged.as_mut() {
            p.owners.park();
        }
        Ok(())
    }

    fn preempt_scheduled_cache(&mut self, seq: SeqId, _: u64, _: PreemptionMode) -> Result<()> {
        let Inner { store, paged, .. } = self;
        let p = paged.as_mut().ok_or_else(no_pool)?;
        if let Some(t) = p.turns.remove(&seq) {
            store.model.clear_prefill_engram_for_prompt(&t.prompt);
        }
        p.lines.remove(&seq);
        if let Some(disk) = p.disk.as_mut() {
            disk.close(seq);
        }
        preempt(&mut p.adapter, seq)
    }

    fn release_scheduled_recurrent_for(&mut self, seq: SeqId) {
        let Inner { store, paged, .. } = self;
        let Some(p) = paged.as_mut() else {
            return;
        };
        if let Some(t) = p.turns.remove(&seq) {
            store.model.clear_prefill_engram_for_prompt(&t.prompt);
        }
        p.lines.remove(&seq);
        if let Some(disk) = p.disk.as_mut() {
            disk.close(seq);
        }
        if p.owners.active() == Some(seq) {
            p.saved.clear();
        }
        p.owners.release(seq);
    }

    /// The wave, then the disk's pass (Flash Next `top_up_disk` after a
    /// batched step).
    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        let out = self.decode_rows(rows);
        if let Some(p) = self.paged.as_mut() {
            p.top_up_disk();
        }
        out
    }

    fn run_scheduled_prefill_slice(
        &mut self,
        seq: SeqId,
        source: &[u32],
        _: &Prefix,
        start: usize,
        end: usize,
        stream: Stream,
        _: bool,
    ) -> Result<Option<MxArray>> {
        self.activate_paged_seq(seq)?;
        let held = self.paged()?.owners.get(seq).map_or(0, OwnerCell::len);
        if held != start || end > source.len() || start >= end {
            return Err(bad(format!(
                "slice {start}..{end} of {} on a cell of {held}",
                source.len()
            )));
        }
        if end < source.len() && !self.store.model.has_prefill_engram(source, start) {
            let mut all_ends = self.grid.ends(start, source);
            if !all_ends.contains(&end) {
                all_ends.push(end);
                all_ends.sort_unstable();
                all_ends.dedup();
            }
            if all_ends.len() >= 2 {
                self.store
                    .model
                    .begin_prefill_engram(source, start, &all_ends);
            }
        }
        let _ctx = StreamContext::new(stream);
        let beside = self.paged()?.busy.others_open(seq);
        let t0 = Instant::now();
        let out = self.prefill_rows(seq, &source[start..end], end == source.len());
        if out.is_err() || end == source.len() {
            self.store.model.clear_prefill_engram_for_prompt(source);
        }
        if beside {
            self.waves
                .stalled(u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX));
        }
        out
    }

    fn replace_cached_token_history(&mut self, history: Vec<u32>) {
        if let Some(p) = self.paged.as_mut() {
            p.saved = history;
        }
    }

    fn install_owner_state(&mut self, seq: SeqId, state: &Vec<u32>) {
        let _ = self.activate_row(seq);
        self.replace_cached_token_history(state.clone());
    }

    fn owner_tokens(state: &Vec<u32>) -> &[u32] {
        state
    }

    fn capture_owner_state(&mut self, _: SeqId) -> Vec<u32> {
        self.paged
            .as_ref()
            .map_or_else(Vec::new, |p| p.saved.clone())
    }

    fn build_scheduled_prefix(
        &self,
        _: &Prefix,
        cached: usize,
        suffix: usize,
        _: Vec<u32>,
        _: bool,
    ) -> Prefix {
        Prefix { cached, suffix }
    }

    /// A whole turn on its owner's cell, landing in the owner tables as a
    /// scheduled one does (Flash Next).
    fn execute_chat_barrier(
        &mut self,
        command: crate::engine::cmd::ChatCmd,
        owners: SchedulerOwnerContext<'_, Vec<u32>>,
    ) {
        crate::engine::hybrid_scheduler::execute_exclusive_owner_barrier(
            self,
            command,
            owners,
            Self::activate_exclusive_seq,
            |inner| {
                inner
                    .paged
                    .as_ref()
                    .map(|p| p.saved.clone())
                    .unwrap_or_default()
            },
            |inner| {
                if let Some(p) = inner.paged.as_mut() {
                    p.owners.park();
                }
            },
        );
    }

    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        HybridStepExecutor::new(self)
    }
}

/// Loads in paged mode get their pool.
/// The paged mode with `serial` off and a wave row cap that holds a padded
/// call, else the flat mode; `disk` serves the mode's stored states.
/// The mode's pool or the flat states over `disk`; a paged load hands its
/// disk back for [`serve`] once its warm-up ran.
pub(crate) fn attach(inner: &mut Inner, serial: bool, disk: Option<Disk>) -> Result<Option<Disk>> {
    if paged_mode(serial, inner.store.row_cap) {
        inner.paged = Some(Paged::open(&inner.store)?);
        return Ok(disk);
    }
    if let Some(disk) = disk {
        inner.flat = Some(FlatTier::new(disk, inner.store.model.cfg().indexer.kpool));
    }
    Ok(None)
}

/// The paged load after its warm-up: the counters start over, the pool's
/// blocks and RAM states go over `held`, a persistent tier's states taken
/// back.
pub(crate) fn serve(inner: &mut Inner, held: Option<Disk>) {
    inner.waves = stats::WaveLog::default();
    inner.served_from = Counters::now(&inner.store.bank, &inner.store.table);
    if let Some(p) = inner.paged.as_mut() {
        p.published = 0;
        p.restored = 0;
        if let Err(e) = spend_cold_skip(&mut p.adapter) {
            tracing::warn!("hy_v4_flash: {}", e.reason);
        }
        if let Some(disk) = held
            && let Err(e) = p.attach_disk(disk)
        {
            tracing::warn!("hy_v4_flash: stored states stay in memory: {}", e.reason);
        }
    }
    inner.publish(None);
}

/// The warm-up's command reset armed the adapter to skip its next cold
/// restore, which would drop the first stored state read back from disk:
/// one empty lookup with no request live takes that skip.
fn spend_cold_skip(adapter: &mut PagedKVCacheAdapter) -> Result<()> {
    if !adapter.live_seq_ids().is_empty() {
        return Ok(());
    }
    adapter.begin_request(0).map_err(bad)?;
    let looked = adapter.find_cached_prefix_per_block(&[], &[], 0, true);
    adapter.release_request_for(0).map_err(bad)?;
    looked.map(|_| ()).map_err(bad)
}
