//! The books and the one rule, after Splash `MemoryLedger`. No MLX and no
//! OS calls: the governor passes the room, the MLX free pool and the
//! sources of the caller's model thread.
//!
//! `free` runs in three steps so that the governor can hold its mutex only
//! around the books: `free_inputs` reads rights and holds, `decide` counts
//! and takes over the caller's sources, `record_free` books the outcome and
//! the hold.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;

use crate::models::qwen3_5::dflash2_state::store::EntryId;
use crate::transformer::paged_kv_cache_adapter::SeqId;

const MIB: f64 = (1u64 << 20) as f64;

/// Splash `MemoryLedger::kCopyDemandSeconds`.
const COPY_DEMAND_SECONDS: f64 = 60.0;

/// Why an allocation is made: decides the host's rule and what `free` may
/// take. Floor (load) is a host rule only and takes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClaimClass {
    Floor,
    InService,
    Start,
    Optional,
    Growth,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NeedTiming {
    CanWait,
    Immediate,
}

/// `InUse`: may take what unfinished requests use (Splash `ReclaimClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReachClass {
    Ordinary,
    InUse,
}

/// Higher gets memory first; compared field by field (Splash `ClaimRank`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ClaimRank {
    pub priority: u8,
    pub decoding: bool,
    pub completed_tokens: u64,
    pub earliness: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClaimSpec {
    pub cls: ClaimClass,
    /// 0: the engine's own, the host.
    pub request_id: u64,
    pub rank: ClaimRank,
    pub reach: ReachClass,
    pub timing: NeedTiming,
}

/// From 1, never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ClaimId(u64);

/// What a refused claim lacks. `bytes` and `host_refused` come from the
/// admission refused inside its scope; `pages` (blocks) is the caller's
/// in-place part.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Need {
    pub bytes: u64,
    pub host_refused: bool,
    pub pages: u32,
    pub kv: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FreeOutcome {
    Met,
    Pending,
    Short,
}

/// What `free` gave, or what one unit's take gave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FreeResult {
    pub outcome: FreeOutcome,
    pub now_bytes: u64,
    /// Leaves memory once transfers it started land.
    pub later_bytes: u64,
    pub pages: u32,
}

impl Default for FreeResult {
    fn default() -> Self {
        Self {
            outcome: FreeOutcome::Short,
            now_bytes: 0,
            later_bytes: 0,
            pages: 0,
        }
    }
}

impl FreeResult {
    /// It gave something: met, or memory, or pages.
    pub(crate) fn gave(&self) -> bool {
        self.outcome == FreeOutcome::Met
            || self.now_bytes > 0
            || self.later_bytes > 0
            || self.pages > 0
    }
}

/// What a source's `take` reports (Splash `VictimSource::take`): Met when
/// it took something, Pending while a transfer in flight holds the unit,
/// Short when nothing of it is left.
pub(crate) type Taken = FreeResult;

/// Splash `MemoryLedger::Order::Stage`. `PassOrConversion` and
/// `StoreAboveFloor` have no mlx-node units, so they are left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Stage {
    FreeNow,
    Rebuild,
    NeedsWrite,
    InUse,
    LaneStart,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum UnitKey {
    Store(EntryId),
    Ladder {
        owner_id: String,
        prefix_len: u32,
        final_block_hash: u64,
    },
    History,
    Owner(SeqId),
    Extent,
    /// The pool's dead leaves, taken until one more extent is cut.
    DeadLeaves,
    Vision(u64),
    /// A disk-only state's chain blocks still to reach the SSD tier.
    StoreChain(EntryId),
}

/// One unit a source would give up. `bytes` and `pages` may over-count
/// (Splash `VictimSource::peek` contract), never under-count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Victim {
    pub key: UnitKey,
    /// Older first within a stage, across sources: the single cache walk
    /// of Splash, whose states and lanes mlx-node keeps in two sources.
    pub stamp: u64,
    pub bytes: u64,
    pub pages: u32,
    pub stage: Stage,
    /// It frees memory only once a copy lands (Splash `Victim::later`): an
    /// Immediate claim passes over it.
    pub later: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CertainReuse {
    #[default]
    None,
    ResumePoint,
    All,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Reserves {
    #[default]
    None,
    NextExtent,
    All,
}

/// What a claim may take (Splash `TakeRights`, restricted to the fields
/// mlx-node has units for). The default is the narrowest: nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TakeRights {
    pub units: bool,
    pub waits_for_writes: bool,
    pub waits_for_landings: bool,
    pub in_use: bool,
    pub certain_reuse: CertainReuse,
    pub reserves: Reserves,
    pub rebuilt_pools: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VictimQuery {
    pub rights: TakeRights,
    pub need: Need,
}

/// A source's units in its take order, head first; a running total of
/// `bytes` / `pages` through a unit bounds what taking every unit up to it
/// frees (Splash `VictimSource::peek` contract). A source leaves out what
/// the query's rights do not take (kept prompt, queued chains, resume
/// point); the ledger leaves out stages the rights exclude.
pub(crate) trait VictimSource {
    fn units(&self, query: &VictimQuery) -> Vec<Victim>;
    fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken;
}

/// A model source's units in Splash's walk phases (`Cache::WalkCursor::Phase`,
/// `Cache::peek` over `ReclaimPhase`): `ahead` (the empty extents, the
/// stored checkpoints no turn uses, oldest first, then the dead leaves),
/// `merged` in stamp order, into which the scheduler merges its idle
/// owners, then `after` (image features, then the states turns use). The
/// source lists them stably sorted by `(stage, later)`, so the phase order
/// holds within each group: a write goes after every unit that frees now
/// and after the image features.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct UnitRuns {
    pub(crate) ahead: Vec<Victim>,
    pub(crate) merged: Vec<Victim>,
    pub(crate) after: Vec<Victim>,
}

impl UnitRuns {
    pub(crate) fn into_units(self) -> Vec<Victim> {
        self.merge_into_units(Vec::new())
    }

    /// The units with `owners` (stamp order) merged into `merged` by
    /// stamp, a state first on a tie.
    pub(crate) fn merge_into_units(self, owners: Vec<Victim>) -> Vec<Victim> {
        let mut merged: Vec<(Victim, bool)> = self
            .merged
            .into_iter()
            .map(|unit| (unit, false))
            .chain(owners.into_iter().map(|unit| (unit, true)))
            .collect();
        merged.sort_by_key(|(unit, owner)| (unit.stamp, *owner));
        let mut units = self.ahead;
        units.extend(merged.into_iter().map(|(unit, _)| unit));
        units.extend(self.after);
        units.sort_by_key(|unit| (unit.stage, unit.later));
        units
    }
}

/// The prompts of requests waiting for a lane (Splash
/// `Cache::setWantedChains`): a claim without certain reuse keeps the
/// states they restore through. Matched by tokens: no salt or image keys.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WantedChains {
    prompts: Vec<Vec<u32>>,
}

impl WantedChains {
    pub(crate) fn new(prompts: Vec<Vec<u32>>) -> Self {
        Self { prompts }
    }

    /// Each waiting prompt as its lookup walks it, the last token left out.
    pub(crate) fn lookups(&self) -> impl Iterator<Item = &[u32]> {
        self.prompts
            .iter()
            .map(|prompt| &prompt[..prompt.len().saturating_sub(1)])
    }

    /// Whether a state on `tokens` is a replay point a waiting prompt's
    /// lookup restores; the lookup leaves the prompt's last token out.
    pub(crate) fn keeps(&self, tokens: &[u32]) -> bool {
        !tokens.is_empty()
            && self
                .prompts
                .iter()
                .any(|prompt| prompt[..prompt.len().saturating_sub(1)].starts_with(tokens))
    }
}

/// What transfers free when they land (Splash `Returning`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Returning {
    pub landing_bytes: u64,
    pub owed_bytes: u64,
    pub landing_pages: u32,
    pub owed_pages: u32,
}

impl Returning {
    fn plus(self, other: Self) -> Self {
        Self {
            landing_bytes: self.landing_bytes.saturating_add(other.landing_bytes),
            owed_bytes: self.owed_bytes.saturating_add(other.owed_bytes),
            landing_pages: self.landing_pages.saturating_add(other.landing_pages),
            owed_pages: self.owed_pages.saturating_add(other.owed_pages),
        }
    }
}

/// Splash `MemoryGovernor::chargedBytes` parts: `allocated` = MLX active +
/// registered pools, `untracked` = the plans' reserves, `cache` = the MLX
/// free pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Charge {
    pub allocated: u64,
    pub untracked: u64,
    pub cache: u64,
}

impl Charge {
    pub(crate) fn charged(&self) -> u64 {
        self.allocated
            .saturating_add(self.untracked)
            .max(self.allocated.saturating_add(self.cache))
    }

    /// What clearing the free pool lowers `charged` by, without clearing:
    /// 0 while the pool sits within the untracked reserve, which charges
    /// those bytes already.
    pub(crate) fn unsized_bytes(&self) -> u64 {
        self.charged()
            .saturating_sub(self.allocated.saturating_add(self.untracked))
    }
}

/// The MLX free pool: the one unit that costs nothing, taken before any
/// count (Splash `MemoryLedger::takeUnsized`).
pub(crate) trait FreePool {
    fn charge(&self) -> Charge;
    fn clear(&mut self);
}

/// Credits the measured drop in charged, not the cache cleared.
fn take_unsized(pool: &mut dyn FreePool) -> u64 {
    let before = pool.charge();
    pool.clear();
    let after = pool.charge();
    before.charged().saturating_sub(after.charged())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LackKind {
    pub bytes: bool,
    pub host_refused: bool,
    pub pages: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Logged {
    pub lack: LackKind,
    pub decisions: u8,
}

/// What the ledger knows of an open claim (Splash `ClaimState`).
#[derive(Clone, Debug)]
pub(crate) struct ClaimState {
    pub spec: ClaimSpec,
    /// The registered model whose pool its page needs and holds are in;
    /// 0: none.
    pub model: u64,
    /// The latest refusal in its scope: bytes and host_refused only.
    pub refusal: Need,
    /// What refused it (Splash `AllocationFailure`).
    pub failure: Option<super::governor::Failure>,
    /// The need of its latest `free` that left it waiting: its hold.
    pub need: Need,
    pub waiting: bool,
    pub waiting_since: f64,
    pub host_refused_waiting: bool,
    pub host_exempt: bool,
    logged: Vec<Logged>,
}

/// Splash `takeRights`. `StoreAboveFloor`, the pass command, lane
/// conversion and the next lane's buffers have no mlx-node units; with no
/// request priority no claim is below a lane.
pub(crate) fn take_rights(spec: &ClaimSpec, need: &Need, critical: bool) -> TakeRights {
    if spec.cls == ClaimClass::Floor {
        panic!("a Floor claim takes nothing: the load passes only the host's rule");
    }
    if spec.cls == ClaimClass::Optional && spec.request_id == 0 {
        return TakeRights::default();
    }
    let host = spec.cls == ClaimClass::Host;
    let in_service = spec.cls == ClaimClass::InService;
    let start = spec.cls == ClaimClass::Start;
    let host_critical = host && critical;
    let waits = if host {
        !critical
    } else {
        spec.timing == NeedTiming::CanWait
    } && !need.host_refused;
    let certain_reuse = if in_service || host_critical {
        CertainReuse::All
    } else if start || spec.cls == ClaimClass::Optional {
        CertainReuse::ResumePoint
    } else {
        CertainReuse::None
    };
    let reserves = if critical {
        Reserves::All
    } else if in_service || start {
        Reserves::NextExtent
    } else {
        Reserves::None
    };
    TakeRights {
        units: true,
        waits_for_writes: waits,
        waits_for_landings: waits && !host,
        in_use: (in_service && waits && spec.reach == ReachClass::InUse) || host_critical,
        certain_reuse,
        reserves,
        rebuilt_pools: spec.cls != ClaimClass::Growth,
    }
}

/// What a claim lacks in each measure that can meet it. `in_place`: a need
/// the host refused with no pages, met by the bytes states drop into the
/// MLX free pool, which the refused allocation reuses in place (Splash
/// state buffers, `Cache::walkFor` `statesOnly`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lacking {
    bytes: u64,
    pages: u32,
    in_place: bool,
}

impl Lacking {
    fn none(&self) -> bool {
        self.bytes == 0 && self.pages == 0
    }

    fn meets(&self, victim: &Victim) -> bool {
        (self.bytes > 0 && victim.bytes > 0) || (self.pages > 0 && victim.pages > 0)
    }

    fn covered(&self, bytes: u64, pages: u32) -> bool {
        (self.bytes > 0 && bytes >= self.bytes) || (self.pages > 0 && pages >= self.pages)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Took {
    pub source: usize,
    pub key: UnitKey,
    pub bytes: u64,
}

/// What the units of a run give: counted (`bound`) or taken (`take`).
#[derive(Clone, Debug)]
pub(crate) struct Run {
    pub took: Vec<Took>,
    /// Now and later.
    pub bytes: u64,
    pub pages: u32,
    pub covered: bool,
    pub freed: FreeResult,
    /// A unit it reached waits for a transfer in flight.
    pub pending: bool,
    /// It reached a unit that frees memory only once its copy lands.
    pub uncopied: bool,
}

impl Default for Run {
    fn default() -> Self {
        Self {
            took: Vec::new(),
            bytes: 0,
            pages: 0,
            covered: false,
            freed: FreeResult {
                outcome: FreeOutcome::Met,
                ..FreeResult::default()
            },
            pending: false,
            uncopied: false,
        }
    }
}

struct Head {
    units: Vec<Victim>,
    at: usize,
}

/// Splash `MemoryLedger::Order`: the lowest stage first; within a stage
/// the oldest stamp, then the source listed first.
struct Order {
    heads: Vec<Head>,
}

impl Order {
    fn new(sources: &[&mut dyn VictimSource], query: &VictimQuery) -> Self {
        Self {
            heads: sources
                .iter()
                .map(|source| Head {
                    units: source.units(query),
                    at: 0,
                })
                .collect(),
        }
    }

    fn admitted(rights: &TakeRights, victim: &Victim) -> bool {
        match victim.stage {
            Stage::Rebuild => rights.rebuilt_pools,
            Stage::InUse => rights.in_use,
            _ => true,
        }
    }

    /// The head to count or take next, units that meet none of the lack
    /// passed over, and with `pass_later` the units that land later (which
    /// set `uncopied`); none above `ceiling`.
    fn best(
        &mut self,
        lack: &Lacking,
        rights: &TakeRights,
        ceiling: Stage,
        pass_later: bool,
        uncopied: &mut bool,
    ) -> Option<usize> {
        let mut best: Option<(Stage, u64, usize)> = None;
        for (index, head) in self.heads.iter_mut().enumerate() {
            while let Some(victim) = head.units.get(head.at) {
                let passed_later = pass_later && victim.later;
                if lack.meets(victim) && Self::admitted(rights, victim) && !passed_later {
                    break;
                }
                *uncopied |= passed_later;
                head.at += 1;
            }
            let Some(victim) = head.units.get(head.at) else {
                continue;
            };
            if victim.stage > ceiling {
                continue;
            }
            let order = (victim.stage, victim.stamp, index);
            if best.is_none_or(|current| order < current) {
                best = Some(order);
            }
        }
        best.map(|(_, _, index)| index)
    }

    fn victim(&self, head: usize) -> &Victim {
        let head = &self.heads[head];
        &head.units[head.at]
    }

    fn pass(&mut self, head: usize) {
        self.heads[head].at += 1;
    }

    /// After a take, the source's units from its head again: what a take
    /// freed can bring back units passed over.
    fn again(&mut self, head: usize, source: &dyn VictimSource, query: &VictimQuery) {
        self.heads[head] = Head {
            units: source.units(query),
            at: 0,
        };
    }
}

/// Splash `MemoryLedger::bound`: the most the order gives, counted without
/// taking, up to the unit that covers the lack. A unit that lands later is
/// reached (`uncopied`) and, for a claim that cannot wait for writes,
/// passed over.
fn bound(sources: &[&mut dyn VictimSource], query: &VictimQuery, lack: &Lacking) -> Run {
    let mut run = Run::default();
    if !query.rights.units {
        return run;
    }
    let mut order = Order::new(sources, query);
    let mut passed = false;
    while let Some(head) = order.best(lack, &query.rights, Stage::LaneStart, false, &mut passed) {
        let victim = order.victim(head);
        if victim.later {
            run.uncopied = true;
            if !query.rights.waits_for_writes {
                order.pass(head);
                continue;
            }
        }
        run.bytes = run.bytes.saturating_add(victim.bytes);
        run.pages = run.pages.saturating_add(victim.pages);
        if lack.covered(run.bytes, run.pages) {
            run.covered = true;
            break;
        }
        order.pass(head);
    }
    run
}

/// Splash `MemoryLedger::take`: the order's units one at a time until what
/// they freed covers the lack. A unit whose take gives nothing is passed
/// over; once a unit is pending for a claim that waits, nothing past the
/// writes is taken.
fn take(sources: &mut [&mut dyn VictimSource], query: &VictimQuery, lack: &Lacking) -> Run {
    let mut order = Order::new(sources, query);
    let mut run = Run::default();
    let mut ceiling = Stage::LaneStart;
    let pass_later = !query.rights.waits_for_writes;
    while !lack.none() && !lack.covered(run.bytes, run.pages) {
        let Some(head) = order.best(lack, &query.rights, ceiling, pass_later, &mut run.uncopied)
        else {
            break;
        };
        let unit = order.victim(head).clone();
        let taken = sources[head].take(query, &unit);
        run.uncopied |= unit.later;
        if !taken.gave() {
            if taken.outcome == FreeOutcome::Pending && query.rights.waits_for_writes {
                run.pending = true;
                ceiling = ceiling.min(unit.stage.max(Stage::NeedsWrite));
            }
            order.pass(head);
            continue;
        }
        if taken.outcome == FreeOutcome::Pending {
            run.freed.outcome = FreeOutcome::Pending;
        }
        run.freed.now_bytes = run.freed.now_bytes.saturating_add(taken.now_bytes);
        run.freed.later_bytes = run.freed.later_bytes.saturating_add(taken.later_bytes);
        run.bytes = run.freed.now_bytes.saturating_add(run.freed.later_bytes);
        run.pages = run.pages.saturating_add(taken.pages);
        let bytes = taken.now_bytes.saturating_add(taken.later_bytes);
        match run.took.last_mut() {
            Some(last) if last.source == head && last.key == unit.key => {
                last.bytes = last.bytes.saturating_add(bytes);
            }
            _ => run.took.push(Took {
                source: head,
                key: unit.key.clone(),
                bytes,
            }),
        }
        order.again(head, &*sources[head], query);
    }
    run.covered = lack.covered(run.bytes, run.pages);
    if run.freed.later_bytes > 0 {
        run.freed.outcome = FreeOutcome::Pending;
    }
    run
}

#[cfg(test)]
/// The room a `free` or `available` asks with: the governor's room under
/// its limit (Host passes 0), macOS's critical pressure, and whether a
/// host-refused need that stays short passes the host (InService).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Room {
    pub bytes: u64,
    pub critical: bool,
    pub exempts: bool,
}

/// What `free` reads from the books before it counts and takes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FreeInputs {
    pub claim: ClaimId,
    pub spec: ClaimSpec,
    pub need: Need,
    pub rights: TakeRights,
    held: u64,
    held_above: u64,
    held_need_pages: u32,
    page_bytes: u64,
    all: Returning,
    own: Returning,
}

impl FreeInputs {
    /// Splash `MemoryLedger::heldPages`: the pages waiting claims ranked
    /// above lack, plus the bytes they hold beyond `room` in pages.
    fn held_pages(&self, room: u64) -> u32 {
        held_pages(self.held_need_pages, self.held_above, room, self.page_bytes)
    }
}

fn held_pages(need_pages: u32, held_bytes: u64, room: u64, page_bytes: u64) -> u32 {
    if page_bytes == 0 {
        return need_pages;
    }
    let pages = held_bytes.saturating_sub(room).div_ceil(page_bytes);
    u64::from(need_pages)
        .saturating_add(pages)
        .min(u64::from(u32::MAX)) as u32
}

/// `free`'s count and take, over the caller's sources.
#[derive(Clone, Debug)]
pub(crate) struct Decision {
    pub result: FreeResult,
    pub run: Option<Run>,
    /// The count or the take reached a unit that lands later (Splash
    /// `MemoryLedger::free`: `most.uncopied`, `run->uncopied`).
    pub uncopied: bool,
}

/// Splash `MemoryLedger::free` without the books: the free pool first;
/// met when the room covers; pending when what lands covers; else nothing
/// is taken unless the count covers the lack, and a claim short only by
/// the room claims ranked above it hold waits.
pub(crate) fn decide(
    inputs: &FreeInputs,
    room_bytes: u64,
    pool: &mut dyn FreePool,
    sources: &mut [&mut dyn VictimSource],
) -> Decision {
    let need = inputs.need;
    let query = VictimQuery {
        rights: inputs.rights,
        need,
    };
    let mut result = FreeResult {
        now_bytes: take_unsized(pool),
        ..FreeResult::default()
    };
    let room = room_bytes.saturating_add(result.now_bytes);
    let held = inputs.held;
    let for_claim = room.saturating_sub(held);
    let lacking_bytes = if need.host_refused || need.bytes == 0 {
        0
    } else {
        need.bytes.saturating_add(held).saturating_sub(room)
    };
    let lacking_pages = need.pages;
    let can_wait = inputs.rights.waits_for_landings;
    let lacks = lacking_bytes > 0 || lacking_pages > 0;
    let (all, own, held_pages_above) = if can_wait && lacks {
        (inputs.all, inputs.own, inputs.held_pages(room))
    } else {
        (Returning::default(), Returning::default(), 0)
    };
    let waits = |owed_too: bool| -> bool {
        if !can_wait || !lacks {
            return false;
        }
        let landing = all.landing_bytes.saturating_add(if owed_too {
            all.owed_bytes
        } else {
            own.owed_bytes
        });
        let landing_pages = u64::from(all.landing_pages)
            + u64::from(if owed_too {
                all.owed_pages
            } else {
                own.owed_pages
            });
        if lacking_bytes > 0
            && (for_claim.saturating_add(landing) >= need.bytes
                || (!owed_too && room.saturating_add(all.landing_bytes) >= need.bytes))
        {
            return true;
        }
        lacking_pages > 0
            && (landing_pages.saturating_sub(u64::from(held_pages_above))
                >= u64::from(lacking_pages)
                || (!owed_too && all.landing_pages >= lacking_pages))
    };
    let credit_bytes = all.landing_bytes.saturating_add(own.owed_bytes);
    let credit_pages = all
        .landing_pages
        .saturating_add(own.owed_pages)
        .saturating_sub(held_pages_above);
    let in_place = need.host_refused && need.pages == 0 && need.bytes > 0;

    let mut run = None;
    let mut uncopied = false;
    if !need.host_refused && lacking_bytes == 0 && (need.bytes > 0 || lacking_pages == 0) {
        result.outcome = FreeOutcome::Met;
    } else if waits(false) {
        result.outcome = FreeOutcome::Pending;
    } else {
        let lack = Lacking {
            bytes: if in_place {
                need.bytes
            } else {
                lacking_bytes.saturating_sub(credit_bytes)
            },
            pages: lacking_pages.saturating_sub(credit_pages),
            in_place,
        };
        let most = bound(sources, &query, &lack);
        uncopied = most.uncopied;
        if most.covered {
            let taken_run = take(sources, &query, &lack);
            uncopied |= taken_run.uncopied;
            let mut taken = taken_run.freed;
            // The dropped buffers land in the MLX free pool, which charged
            // still counts, unless they meet the need in place.
            if lack.bytes > 0 && !lack.in_place && !taken_run.took.is_empty() {
                taken.now_bytes = taken.now_bytes.saturating_add(take_unsized(pool));
            }
            let alone = (lacking_bytes > 0 && taken_run.bytes >= lacking_bytes)
                || (lacking_pages > 0 && taken_run.pages >= lacking_pages)
                || (in_place && taken_run.bytes >= need.bytes);
            let landing = taken_run.pending || taken.later_bytes > 0 || waits(true);
            let outcome = if !taken_run.covered {
                if landing {
                    FreeOutcome::Pending
                } else {
                    FreeOutcome::Short
                }
            } else if alone {
                taken.outcome
            } else {
                FreeOutcome::Pending
            };
            result = FreeResult {
                outcome,
                now_bytes: result.now_bytes.saturating_add(taken.now_bytes),
                later_bytes: taken.later_bytes,
                pages: taken_run.pages,
            };
            run = Some(taken_run);
        } else {
            // Short would end a lone lane as if at its capacity: short only
            // by the room held above, it waits and takes nothing.
            let held_above = can_wait
                && lacking_bytes > 0
                && held > 0
                && most.bytes.saturating_add(credit_bytes).saturating_add(held) >= lacking_bytes;
            result.outcome = if waits(true) || held_above {
                FreeOutcome::Pending
            } else {
                FreeOutcome::Short
            };
        }
    }
    Decision {
        result,
        run,
        uncopied,
    }
}

/// What `count_available` reads from the books.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AvailableInputs {
    query: VictimQuery,
    held: u64,
    landing: u64,
}

/// Splash `MemoryLedger::available` without the books: `free` counted,
/// taking nothing. The free pool counts as the drop clearing it would
/// give.
pub(crate) fn count_available(
    inputs: &AvailableInputs,
    room_bytes: u64,
    charge: Charge,
    sources: &[&mut dyn VictimSource],
) -> u64 {
    let enough = inputs.query.need.bytes;
    let held = inputs.held;
    let room = room_bytes
        .saturating_add(charge.unsized_bytes())
        .saturating_add(inputs.landing);
    if room.saturating_sub(held) >= enough {
        return enough;
    }
    let lack = Lacking {
        bytes: enough.saturating_add(held).saturating_sub(room),
        pages: 0,
        in_place: false,
    };
    let taken = bound(sources, &inputs.query, &lack).bytes;
    enough.min(room.saturating_add(taken).saturating_sub(held))
}

type AccountFn = Arc<dyn Fn() -> u64 + Send + Sync>;
type ReturningFn = Arc<dyn Fn() -> Returning + Send + Sync>;
type ReturningForFn = Arc<dyn Fn(u64) -> Returning + Send + Sync>;
type LogFn = Arc<dyn Fn(&str) + Send + Sync>;

struct Account {
    model: u64,
    name: &'static str,
    current: AccountFn,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AccountRow {
    pub name: String,
    pub bytes: u64,
}

/// Splash `MemoryLedger::fill`: the accounts summed by name, "fixed" first
/// (charged less the others, so the accounts sum to charged), what lands,
/// and the waiting claims.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LedgerSnapshot {
    pub accounts: Vec<AccountRow>,
    pub owed_bytes: u64,
    pub owed_to_claims_bytes: u64,
    pub claims_waiting: u32,
    pub claims_bytes: u64,
    pub oldest_waiting_ms: f64,
}

pub(crate) struct Ledger {
    claims: BTreeMap<u64, ClaimState>,
    next_claim: u64,
    accounts: Vec<Account>,
    page_bytes: BTreeMap<u64, u64>,
    returning: Option<ReturningFn>,
    returning_for: Option<ReturningForFn>,
    /// Each model's transfers in flight (Splash `Cache::returningOf`).
    returning_models: Vec<(u64, ReturningFn)>,
    log: Option<LogFn>,
    now: f64,
    /// Splash `MemoryLedger::poolReleases`: pool extents taken.
    pool_releases: u64,
    /// When a `free` last reached a unit that frees memory only once its
    /// copy lands (Splash `uncopiedReachedSeconds_`).
    uncopied_reached: Option<f64>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

impl Ledger {
    pub(crate) fn new() -> Self {
        Self {
            claims: BTreeMap::new(),
            next_claim: 1,
            accounts: Vec::new(),
            page_bytes: BTreeMap::new(),
            returning: None,
            returning_for: None,
            returning_models: Vec::new(),
            log: None,
            now: 0.0,
            pool_releases: 0,
            uncopied_reached: None,
        }
    }

    pub(crate) fn pool_releases(&self) -> u64 {
        self.pool_releases
    }

    /// Splash `MemoryLedger::copyDemand`: a `free` reached a unit that frees
    /// memory only once its copy lands within the last
    /// [`COPY_DEMAND_SECONDS`].
    pub(crate) fn copy_demand(&self, now: f64) -> bool {
        self.uncopied_reached
            .is_some_and(|reached| now - reached <= COPY_DEMAND_SECONDS)
    }

    /// The control clock waits are dated by.
    pub(crate) fn set_now(&mut self, seconds: f64) {
        self.now = seconds;
    }

    /// The bytes of one KV block of `model`'s pool, in which held bytes are
    /// held as pages.
    pub(crate) fn set_page_bytes(&mut self, model: u64, bytes: u64) {
        self.page_bytes.insert(model, bytes);
    }

    #[cfg(test)]
    /// What transfers return when they land: all of them, and one
    /// request's owed copies.
    pub(crate) fn set_returning(&mut self, all: ReturningFn, of_request: ReturningForFn) {
        self.returning = Some(all);
        self.returning_for = Some(of_request);
    }

    pub(crate) fn set_decision_log(&mut self, log: LogFn) {
        self.log = Some(log);
    }

    /// An account pulled from its owner at each `fill`; accounts of one
    /// name are summed across models.
    pub(crate) fn add_account(&mut self, model: u64, name: &'static str, current: AccountFn) {
        self.accounts.push(Account {
            model,
            name,
            current,
        });
    }

    /// What `model`'s transfers in flight free when they land, pulled at
    /// each decision.
    pub(crate) fn add_returning(&mut self, model: u64, returning: ReturningFn) {
        self.returning_models.retain(|(owner, _)| *owner != model);
        self.returning_models.push((model, returning));
    }

    /// A model's accounts and page size leave the books.
    pub(crate) fn remove_model(&mut self, model: u64) {
        self.accounts.retain(|account| account.model != model);
        self.returning_models.retain(|(owner, _)| *owner != model);
        self.page_bytes.remove(&model);
    }

    pub(crate) fn open(&mut self, spec: ClaimSpec, model: u64) -> ClaimId {
        let id = self.next_claim;
        self.next_claim += 1;
        self.claims.insert(
            id,
            ClaimState {
                spec,
                model,
                refusal: Need::default(),
                failure: None,
                need: Need::default(),
                waiting: false,
                waiting_since: 0.0,
                host_refused_waiting: false,
                host_exempt: false,
                logged: Vec::new(),
            },
        );
        ClaimId(id)
    }

    pub(crate) fn update(&mut self, claim: ClaimId, spec: ClaimSpec) {
        if let Some(state) = self.claims.get_mut(&claim.0) {
            state.spec = spec;
        }
    }

    /// The latest refusal in the claim's scope; its cause is unknown until
    /// `set_failure`.
    pub(crate) fn refused(&mut self, claim: ClaimId, need: Need) {
        if let Some(state) = self.claims.get_mut(&claim.0) {
            state.refusal = Need {
                bytes: need.bytes,
                host_refused: need.host_refused,
                ..Need::default()
            };
            state.failure = None;
        }
    }

    pub(crate) fn set_failure(&mut self, claim: ClaimId, failure: super::governor::Failure) {
        if let Some(state) = self.claims.get_mut(&claim.0) {
            state.failure = Some(failure);
        }
    }

    pub(crate) fn set_waiting(&mut self, claim: ClaimId, waiting: bool) {
        let now = self.now;
        if let Some(state) = self.claims.get_mut(&claim.0) {
            if waiting && !state.waiting {
                state.waiting_since = now;
            }
            state.waiting = waiting;
        }
    }

    pub(crate) fn set_host_refused_waiting(&mut self, claim: ClaimId, waiting: bool) {
        if let Some(state) = self.claims.get_mut(&claim.0) {
            state.host_refused_waiting = waiting;
        }
    }

    pub(crate) fn set_host_exempt(&mut self, claim: ClaimId, exempt: bool) {
        if let Some(state) = self.claims.get_mut(&claim.0) {
            state.host_exempt = exempt;
        }
    }

    /// An allocation of the claim was granted: what it lacks next logs
    /// afresh.
    pub(crate) fn granted(&mut self, claim: ClaimId) {
        if let Some(state) = self.claims.get_mut(&claim.0) {
            state.logged.clear();
        }
    }

    pub(crate) fn close(&mut self, claim: ClaimId) {
        self.claims.remove(&claim.0);
    }

    pub(crate) fn claim(&self, claim: ClaimId) -> Option<&ClaimState> {
        self.claims.get(&claim.0)
    }

    #[cfg(test)]
    pub(crate) fn claims(&self) -> impl Iterator<Item = (ClaimId, &ClaimState)> {
        self.claims.iter().map(|(id, state)| (ClaimId(*id), state))
    }

    fn held_by_claims(&self, rank: &ClaimRank, own: Option<u64>) -> u64 {
        self.claims
            .iter()
            .filter(|(id, state)| state.waiting && Some(**id) != own && state.spec.rank > *rank)
            .fold(0u64, |held, (_, state)| {
                held.saturating_add(state.need.bytes)
            })
    }

    /// Room under the limit returns nothing to the host: nothing held
    /// stands against its claim.
    fn held_for(&self, spec: &ClaimSpec, own: Option<u64>) -> u64 {
        if spec.cls == ClaimClass::Host {
            return 0;
        }
        self.held_by_claims(&spec.rank, own)
    }

    /// What admission holds against an allocation in `claim`'s scope:
    /// waiting claims ranked above it, process-wide (one device).
    pub(crate) fn held_bytes(&self, claim: ClaimId) -> u64 {
        self.claims
            .get(&claim.0)
            .map_or(0, |state| self.held_for(&state.spec, Some(claim.0)))
    }

    fn held_need_pages(&self, claim: u64, state: &ClaimState) -> u32 {
        self.claims
            .iter()
            .filter(|(id, other)| {
                other.waiting
                    && **id != claim
                    && other.model == state.model
                    && other.spec.rank > state.spec.rank
            })
            .fold(0u32, |pages, (_, other)| {
                pages.saturating_add(other.need.pages)
            })
    }

    /// The free blocks of `claim`'s model held against a block handed out
    /// in its scope: the pages waiting claims of that model ranked above
    /// it lack, plus the bytes they hold beyond `room` in blocks.
    pub(crate) fn held_pages(&self, claim: ClaimId, room: u64) -> u32 {
        let Some(state) = self.claims.get(&claim.0) else {
            return 0;
        };
        held_pages(
            self.held_need_pages(claim.0, state),
            self.held_by_claims(&state.spec.rank, Some(claim.0)),
            room,
            self.page_bytes.get(&state.model).copied().unwrap_or(0),
        )
    }

    pub(crate) fn claim_waiting(&self) -> bool {
        self.claims.values().any(|state| state.waiting)
    }

    /// The bytes the host refused the claims that wait, or whose requests
    /// wait to retry.
    pub(crate) fn host_refused_waiting_bytes(&self) -> u64 {
        self.claims
            .values()
            .filter(|state| {
                (state.waiting || state.host_refused_waiting) && state.refusal.host_refused
            })
            .fold(0u64, |bytes, state| {
                bytes.saturating_add(state.refusal.bytes)
            })
    }

    pub(crate) fn returning(&self) -> Returning {
        self.returning_models.iter().fold(
            self.returning
                .as_ref()
                .map_or_else(Returning::default, |all| all()),
            |sum, (_, returning)| sum.plus(returning()),
        )
    }

    fn returning_of(&self, request_id: u64) -> Returning {
        match &self.returning_for {
            Some(of_request) if request_id != 0 => of_request(request_id),
            _ => Returning::default(),
        }
    }

    /// The books `decide` reads.
    pub(crate) fn free_inputs(&self, claim: ClaimId, need: Need, critical: bool) -> FreeInputs {
        let state = self
            .claims
            .get(&claim.0)
            .unwrap_or_else(|| panic!("memory claim {} is not open", claim.0));
        let rights = take_rights(&state.spec, &need, critical);
        FreeInputs {
            claim,
            spec: state.spec,
            need,
            rights,
            held: self.held_for(&state.spec, Some(claim.0)),
            held_above: self.held_by_claims(&state.spec.rank, Some(claim.0)),
            held_need_pages: self.held_need_pages(claim.0, state),
            page_bytes: self.page_bytes.get(&state.model).copied().unwrap_or(0),
            all: self.returning(),
            own: self.returning_of(state.spec.request_id),
        }
    }

    /// Books `decide`'s outcome: Pending holds the need, Short ends the
    /// wait, Met leaves it (a need met for one allocation does not end the
    /// request's wait for another). A host-refused need that stays short
    /// passes the host when `exempts`.
    pub(crate) fn record_free(
        &mut self,
        inputs: &FreeInputs,
        decision: Decision,
        exempts: bool,
    ) -> FreeResult {
        let mut result = decision.result;
        let need = inputs.need;
        let now = self.now;
        if decision.uncopied {
            self.uncopied_reached = Some(now);
        }
        if let Some(run) = &decision.run {
            let released = run
                .took
                .iter()
                .filter(|took| {
                    matches!(took.key, UnitKey::Extent | UnitKey::DeadLeaves) && took.bytes > 0
                })
                .count() as u64;
            self.pool_releases = self.pool_releases.saturating_add(released);
        }
        let Some(claim) = self.claims.get_mut(&inputs.claim.0) else {
            return result;
        };
        match result.outcome {
            FreeOutcome::Pending => {
                if !claim.waiting {
                    claim.waiting_since = now;
                }
                claim.waiting = true;
                claim.need = need;
            }
            FreeOutcome::Short => claim.waiting = false,
            FreeOutcome::Met => {}
        }
        let exempted = exempts && need.host_refused && result.outcome == FreeOutcome::Short;
        if exempted {
            claim.host_exempt = true;
            result.outcome = FreeOutcome::Met;
        }
        // One line per kind of lack and decision since the claim's last
        // grant; a claim of no request logs each take.
        let mut logs = decision.run.is_some();
        if claim.spec.request_id != 0 {
            let bit = 1u8 << if exempted { 3 } else { result.outcome as u8 };
            let lack = LackKind {
                bytes: need.bytes != 0,
                host_refused: need.host_refused,
                pages: need.pages != 0,
            };
            let index = match claim.logged.iter().position(|logged| logged.lack == lack) {
                Some(index) => index,
                None => {
                    claim.logged.push(Logged { lack, decisions: 0 });
                    claim.logged.len() - 1
                }
            };
            let logged = &mut claim.logged[index];
            logs = logged.decisions & bit == 0;
            logged.decisions |= bit;
        }
        if logs && let Some(log) = &self.log {
            log(&decision_line(
                &inputs.spec,
                &need,
                decision.run.as_ref(),
                &result,
                exempted,
            ));
        }
        result
    }

    #[cfg(test)]
    /// Splash `MemoryLedger::free`, all three steps on the caller's
    /// thread.
    pub(crate) fn free(
        &mut self,
        claim: ClaimId,
        need: Need,
        room: Room,
        pool: &mut dyn FreePool,
        sources: &mut [&mut dyn VictimSource],
    ) -> FreeResult {
        let inputs = self.free_inputs(claim, need, room.critical);
        let decision = decide(&inputs, room.bytes, pool, sources);
        self.record_free(&inputs, decision, room.exempts)
    }

    /// What `count_available` reads from the books.
    pub(crate) fn available_inputs(
        &self,
        spec: &ClaimSpec,
        need: Need,
        critical: bool,
    ) -> AvailableInputs {
        let query = VictimQuery {
            rights: take_rights(spec, &need, critical),
            need,
        };
        let landing = if query.rights.waits_for_landings {
            self.returning()
                .landing_bytes
                .saturating_add(self.returning_of(spec.request_id).owed_bytes)
        } else {
            0
        };
        AvailableInputs {
            query,
            held: self.held_for(spec, None),
            landing,
        }
    }

    #[cfg(test)]
    /// Splash `MemoryLedger::available` on one thread.
    pub(crate) fn available(
        &self,
        spec: &ClaimSpec,
        need: Need,
        room: Room,
        charge: Charge,
        sources: &[&mut dyn VictimSource],
    ) -> u64 {
        count_available(
            &self.available_inputs(spec, need, room.critical),
            room.bytes,
            charge,
            sources,
        )
    }

    /// `measured`: accounts the caller read before taking the books' lock
    /// (the MLX free pool).
    pub(crate) fn fill(&self, charged: u64, measured: &[(&'static str, u64)]) -> LedgerSnapshot {
        let mut rows: Vec<AccountRow> = Vec::new();
        let pulled = self
            .accounts
            .iter()
            .map(|account| (account.name, (account.current)()));
        for (name, bytes) in pulled.chain(measured.iter().copied()) {
            match rows.iter_mut().find(|row| row.name == name) {
                Some(row) => row.bytes = row.bytes.saturating_add(bytes),
                None => rows.push(AccountRow {
                    name: name.to_string(),
                    bytes,
                }),
            }
        }
        let others = rows
            .iter()
            .fold(0u64, |sum, row| sum.saturating_add(row.bytes));
        let fixed = charged.saturating_sub(others);
        let mut accounts = Vec::with_capacity(rows.len() + 1);
        accounts.push(AccountRow {
            name: "fixed".to_string(),
            bytes: fixed,
        });
        accounts.extend(rows);

        let back = self.returning();
        let owed_bytes = back.landing_bytes.saturating_add(back.owed_bytes);
        let mut claims_waiting = 0u32;
        let mut claims_bytes = 0u64;
        let mut oldest: Option<f64> = None;
        for state in self.claims.values().filter(|state| state.waiting) {
            claims_waiting += 1;
            claims_bytes = claims_bytes.saturating_add(state.need.bytes);
            oldest = Some(oldest.map_or(state.waiting_since, |o| o.min(state.waiting_since)));
        }
        LedgerSnapshot {
            accounts,
            owed_bytes,
            owed_to_claims_bytes: owed_bytes.min(claims_bytes),
            claims_waiting,
            claims_bytes,
            oldest_waiting_ms: oldest.map_or(0.0, |since| (self.now - since).max(0.0) * 1000.0),
        }
    }
}

fn class_name(cls: ClaimClass) -> &'static str {
    match cls {
        ClaimClass::Floor => "floor",
        ClaimClass::InService => "in-service",
        ClaimClass::Start => "start",
        ClaimClass::Optional => "optional",
        ClaimClass::Growth => "growth",
        ClaimClass::Host => "host",
    }
}

fn outcome_name(outcome: FreeOutcome) -> &'static str {
    match outcome {
        FreeOutcome::Met => "met",
        FreeOutcome::Pending => "pending",
        FreeOutcome::Short => "short",
    }
}

fn decision_line(
    spec: &ClaimSpec,
    need: &Need,
    run: Option<&Run>,
    result: &FreeResult,
    exempted: bool,
) -> String {
    let mut line = format!("Memory: {} claim", class_name(spec.cls));
    if spec.request_id != 0 {
        let _ = write!(line, " of request {}", spec.request_id);
    }
    let _ = write!(line, " needs {:.1} MiB", need.bytes as f64 / MIB);
    if need.pages != 0 {
        let _ = write!(line, ", {} blocks", need.pages);
    }
    if need.host_refused {
        line.push_str(" the host refused");
    }
    let _ = write!(line, ": {}", outcome_name(result.outcome));
    if exempted {
        line.push_str(" past the host");
    }
    if let Some(run) = run {
        line.push_str(", took");
        for took in &run.took {
            let _ = write!(line, " {:?} {:.1}", took.key, took.bytes as f64 / MIB);
        }
    }
    let _ = write!(
        line,
        " ({:.1} MiB now, {:.1} later)",
        result.now_bytes as f64 / MIB,
        result.later_bytes as f64 / MIB
    );
    line
}

#[cfg(test)]
mod tests;
