//! The move of a held source store's chats into the load's own store, at
//! load, before the store's states are taken back. A source object goes
//! only once the object that holds its rows landed in the own store, read
//! back through the restore's decode and reached the device, for every page
//! made of it (two chats that part inside a page have a page each on its
//! first block); the last block of an odd state stays until the source
//! directory goes; the directory goes only when no state the old load takes
//! back is left in it. Every stop keeps what is left where it is: the next
//! load starts again and skips what the own store holds.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdGroup,
    ColdSourceBlock, ColdSourceReader, ColdStateLabel, ColdStateLayout, ColdStateObject,
    DEFAULT_QUEUE_DEPTH, Landed, MakeRoom, state_object_bytes,
};

use crate::disk_tier::DiskTier;

use super::super::disk::lineage_keys;
use super::super::store::{Lineage, PART_PAGE_ROWS};
use super::{
    Fingerprints, FragmentPlan, PAGE_BLOCK, PagePlan, SOURCE_BLOCK, StatePlan, convert_fragment,
    convert_page, convert_state, page_layout, part_page_keys, plan_state, tail_keys,
};

const _: () = assert!(PART_PAGE_ROWS == SOURCE_BLOCK as usize);

/// How long one write may take to land; past it the move stops for this
/// load and nothing more leaves the source.
const LANDING_WAIT: Duration = Duration::from_secs(60);

/// Source blocks whose rows the own store holds too, at most, before the
/// device is flushed and they go: the bytes on disk twice stay within the
/// writer's queue of pages.
const UNLINK_BATCH: usize = 2 * DEFAULT_QUEUE_DEPTH;

/// What the move needs of the load besides its tier.
pub(crate) struct MoveShape<'a> {
    /// The spans and geometry text of the load's state objects.
    pub(crate) spans: &'a [u64],
    pub(crate) geometry: &'a str,
    /// The layout of the source store's KV blocks ([`super::source_layout`]).
    pub(crate) source_layout: ColdCacheLayout,
}

/// What one run of the move did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Moved {
    /// States whose object is in the own store and left the source.
    pub(crate) states: u64,
    /// Pages and fragments this run wrote.
    pub(crate) pages: u64,
    pub(crate) fragments: u64,
    /// States past the own store's limit: they go with the source.
    pub(crate) refused: u64,
    /// States one of whose source objects is absent or damaged: no load
    /// takes them back.
    pub(crate) unreadable: u64,
    /// States the old load takes back and this move cannot convert, and
    /// source objects that are not what the move takes them for: the source
    /// stays, whole from the first one on.
    pub(crate) held_back: u64,
    /// Writes the writer's queue did not take.
    pub(crate) dropped: u64,
    /// The run stopped early; what it did not move is still in the source.
    pub(crate) stopped: bool,
    pub(crate) source_removed: bool,
}

struct SourceObject {
    bytes: u64,
    modified: u128,
}

struct SourceState {
    key: ColdCacheKey,
    modified: u128,
    layout: ColdStateLayout,
    label: ColdStateLabel,
}

/// The run ends here; everything not yet moved stays in the source.
struct Stop;

enum Outcome {
    Moved,
    Refused,
    Unreadable,
}

enum Standing {
    Convertible(Box<StatePlan>),
    HeldBack,
    Not,
}

#[cfg(test)]
fn cut(step: hooks::Step) -> bool {
    hooks::at(step) == hooks::Act::Cut
}

struct Move<'a> {
    tier: &'a DiskTier,
    reader: ColdSourceReader,
    shape: &'a MoveShape<'a>,
    page_layout: ColdCacheLayout,
    fingerprints: Fingerprints,
    source_state: ColdCacheFingerprint,
    /// The source's KV blocks still in it.
    blocks: HashMap<ColdCacheKey, SourceObject>,
    /// The last block of every odd source state.
    tails: HashSet<ColdCacheKey>,
    /// Per source block, the pages of the states this run moves that are
    /// made of it. Two chats that part inside a page share its first block
    /// and have a page each.
    pages_of: HashMap<ColdCacheKey, Vec<ColdCacheKey>>,
    /// The pages of the own store this run read back.
    read_back: HashSet<ColdCacheKey>,
    /// Own objects no state's move may evict: those of the states this run
    /// handled, all newer than the one in hand, and of the own store's
    /// states at least as new as it.
    keep: HashSet<ColdCacheKey>,
    /// The own store's states, oldest first, each with its time and its
    /// objects' keys; read once, when room first runs short.
    own_states: Option<Vec<(u128, Vec<ColdCacheKey>)>>,
    /// Source objects whose rows the own store holds and read back, not yet
    /// flushed to the device.
    pending: Vec<(ColdCacheKey, ColdGroup)>,
    moved: Moved,
}

/// Moves the chats of the source store `tier` holds into `tier`: newest
/// state first, each state's pages in order, then its fragment, then its
/// state object. A run that ends with no state the old load takes back
/// left in the source removes the source directory. Nothing happens for a
/// tier that holds no source.
pub(crate) fn move_source(tier: &DiskTier, shape: &MoveShape<'_>) -> Moved {
    if !tier.persistent {
        return Moved::default();
    }
    let Some((dir, source_state, source_kv)) =
        tier.with_source(|hold| (hold.dir.clone(), hold.identity.state, hold.identity.kv))
    else {
        return Moved::default();
    };
    let Ok(reader) = ColdSourceReader::open(&dir) else {
        return Moved::default();
    };
    let Ok(entries) = reader.entries() else {
        return Moved::default();
    };
    let mut blocks = HashMap::new();
    let mut states = Vec::new();
    for entry in entries {
        match entry.group {
            ColdGroup::Kv => {
                blocks.insert(
                    entry.key,
                    SourceObject {
                        bytes: entry.bytes,
                        modified: entry.modified,
                    },
                );
            }
            ColdGroup::DFlash2State => {
                if let Some((layout, label)) = reader.state_label(entry.key, source_state) {
                    states.push(SourceState {
                        key: entry.key,
                        modified: entry.modified,
                        layout,
                        label,
                    });
                }
            }
            _ => {}
        }
    }
    let tails = tail_keys(states.iter().map(|state| &state.label), source_kv);
    states.sort_by_key(|state| std::cmp::Reverse((state.label.last_used, state.modified)));
    let mut run = Move {
        tier,
        reader,
        shape,
        page_layout: page_layout(&shape.source_layout),
        fingerprints: Fingerprints {
            source_kv,
            kv: tier.kv_fingerprint,
            state: tier.state_fingerprint,
        },
        source_state,
        blocks,
        tails,
        pages_of: HashMap::new(),
        read_back: HashSet::new(),
        keep: HashSet::new(),
        own_states: None,
        pending: Vec::new(),
        moved: Moved::default(),
    };
    let mut work: Vec<(&SourceState, StatePlan)> = Vec::new();
    for state in &states {
        match run.standing(state) {
            Standing::Convertible(plan) => work.push((state, *plan)),
            Standing::HeldBack => run.moved.held_back += 1,
            Standing::Not => {}
        }
    }
    // No page of this run accounts for the blocks of a state the old load
    // takes back and the move cannot convert: with one, nothing moves.
    run.moved.stopped = run.moved.held_back > 0;
    for (_, plan) in &work {
        for page in &plan.pages {
            for source in page.sources {
                let pages = run.pages_of.entry(source).or_default();
                if !pages.contains(&page.key) {
                    pages.push(page.key);
                }
            }
        }
    }
    let mut plans: Vec<&StatePlan> = Vec::new();
    for (state, plan) in &work {
        if run.moved.stopped {
            break;
        }
        match run.move_state(state, plan) {
            Ok(Outcome::Moved) => {
                run.moved.states += 1;
                plans.push(plan);
            }
            Ok(Outcome::Refused) => run.moved.refused += 1,
            Ok(Outcome::Unreadable) => run.moved.unreadable += 1,
            Err(Stop) => run.moved.stopped = true,
        }
    }
    if !run.moved.stopped && run.settle().is_err() {
        run.moved.stopped = true;
    }
    if !run.moved.stopped && run.moved.held_back == 0 {
        run.moved.source_removed = run.remove_source(&plans);
    }
    let moved = run.moved;
    tracing::info!(
        "SSD cache: moved {} states ({} pages, {} fragments) from {}: {} refused, {} unreadable, \
         {} held back, {} dropped{}{}",
        moved.states,
        moved.pages,
        moved.fragments,
        dir.display(),
        moved.refused,
        moved.unreadable,
        moved.held_back,
        moved.dropped,
        if moved.stopped {
            "; stopped, the next load goes on"
        } else {
            ""
        },
        if moved.source_removed {
            "; the source is removed"
        } else {
            ""
        },
    );
    moved
}

impl Move<'_> {
    /// Whether the old load takes `state` back and how it converts: its
    /// object is of the load's shape and under its own key, and its rows
    /// are whole ([`Self::whole`]).
    fn standing(&self, state: &SourceState) -> Standing {
        let rows = state.label.tokens.len();
        let lineage = Lineage {
            tokens: &state.label.tokens,
            extra_keys: &state.label.extra_keys,
            salt: state.label.salt,
        };
        let own = lineage_keys(
            ColdGroup::DFlash2State,
            self.source_state,
            lineage,
            SOURCE_BLOCK,
        );
        if state.layout.spans != self.shape.spans
            || state.layout.geometry != self.shape.geometry
            || own.len() * SOURCE_BLOCK as usize != rows
            || own.last() != Some(&state.key)
            || u32::try_from(rows).is_err()
        {
            return Standing::Not;
        }
        let Some(plan) = plan_state(&state.label, &self.fingerprints) else {
            let chain = lineage_keys(
                ColdGroup::Kv,
                self.fingerprints.source_kv,
                lineage,
                SOURCE_BLOCK,
            );
            return if chain.iter().all(|key| self.blocks.contains_key(key)) {
                Standing::HeldBack
            } else {
                Standing::Not
            };
        };
        if self.whole(&plan) {
            Standing::Convertible(Box::new(plan))
        } else {
            Standing::Not
        }
    }

    /// Each page of `plan` is in the own store or has both halves in the
    /// source; its fragment is in the own store or has its tail block in
    /// the source.
    fn whole(&self, plan: &StatePlan) -> bool {
        let manager = &self.tier.manager;
        let here = |key: &ColdCacheKey| manager.contains_in(key, ColdGroup::Kv);
        plan.pages.iter().all(|page| {
            here(&page.key) || page.sources.iter().all(|key| self.blocks.contains_key(key))
        }) && plan.fragment.as_ref().is_none_or(|fragment| {
            here(&fragment.key) || self.blocks.contains_key(&fragment.source)
        })
    }

    /// The page or fragment for `key` is in the own store: it is not made
    /// again.
    fn on_tier(&self, key: &ColdCacheKey) -> bool {
        self.tier.manager.contains_in(key, ColdGroup::Kv)
    }

    /// Source block `key` is some state's last block past its last whole
    /// page: that state's fragment is made of it, so it stays until the
    /// source directory goes, whatever page of a longer chat holds its rows.
    fn stays(&self, key: &ColdCacheKey) -> bool {
        self.tails.contains(key)
    }

    /// Every page of this run made of source block `key` is in the own
    /// store and read back: no state of the run still needs the block.
    fn free_to_go(&self, key: &ColdCacheKey) -> bool {
        self.pages_of
            .get(key)
            .is_none_or(|pages| pages.iter().all(|page| self.read_back.contains(page)))
    }

    /// The source is not what this load takes it for: the run ends and the
    /// source stays.
    fn hold(&mut self) -> Stop {
        self.moved.held_back += 1;
        Stop
    }

    /// A moved object takes its source's time, so the own store evicts in
    /// the source's order.
    fn keep_time(&self, key: &ColdCacheKey, time: Option<u128>) {
        if let Some(time) = time {
            self.tier.manager.set_modified(key, time);
        }
    }

    /// An own object this run did not write takes its source's time only
    /// when that is the newer one: a use of it since is not undone.
    fn raise_time(&self, key: &ColdCacheKey, time: Option<u128>) {
        let own = self.tier.manager.modified(key);
        if let Some(time) = time.filter(|time| own.is_none_or(|own| own < *time)) {
            self.keep_time(key, Some(time));
        }
    }

    /// Room in the own store for `needed` more bytes of a state last used
    /// at `used`: free room, else what own objects older than the state
    /// give up. No newer object goes for an older state.
    fn make_room(&mut self, needed: u64, used: u128) -> bool {
        let tier = self.tier;
        let manager = &tier.manager;
        if manager.total_bytes().saturating_add(needed) <= manager.quota_bytes() {
            return true;
        }
        let own = self.own_states.get_or_insert_with(|| {
            let mut own: Vec<(u128, Vec<ColdCacheKey>)> = manager
                .keys_in(ColdGroup::DFlash2State)
                .into_iter()
                .filter_map(|key| {
                    let at = manager.modified(&key)?;
                    let (_, label) = manager.state_label(key, tier.state_fingerprint)?;
                    let lineage = Lineage {
                        tokens: &label.tokens,
                        extra_keys: &label.extra_keys,
                        salt: label.salt,
                    };
                    let (mut keys, part) =
                        part_page_keys(ColdGroup::Kv, tier.kv_fingerprint, lineage, PAGE_BLOCK);
                    keys.extend(part);
                    keys.push(key);
                    Some((at, keys))
                })
                .collect();
            own.sort_unstable_by_key(|(at, _)| *at);
            own
        });
        while let Some((_, keys)) = own.pop_if(|(at, _)| *at >= used) {
            self.keep.extend(keys);
        }
        manager.make_room_among_older(needed, used, &self.keep)
    }

    fn move_state(&mut self, state: &SourceState, plan: &StatePlan) -> Result<Outcome, Stop> {
        let tier = self.tier;
        let manager = &tier.manager;
        // An own page of it gave way to a newer state since the scan: it is
        // past the limit.
        if !self.whole(plan) {
            return Ok(Outcome::Refused);
        }
        let label = ColdStateLabel {
            extra_keys: plan.extra_keys.clone(),
            ..state.label.clone()
        };
        let source_bytes = |key: &ColdCacheKey| self.blocks.get(key).map_or(0, |block| block.bytes);
        let mut needed = 0u64;
        for page in &plan.pages {
            if !manager.contains_in(&page.key, ColdGroup::Kv) {
                needed = page
                    .sources
                    .iter()
                    .fold(needed, |sum, key| sum.saturating_add(source_bytes(key)));
            }
        }
        if let Some(fragment) = &plan.fragment
            && !manager.contains_in(&fragment.key, ColdGroup::Kv)
        {
            needed = needed.saturating_add(source_bytes(&fragment.source));
        }
        if !manager.contains_in(&plan.state_key, ColdGroup::DFlash2State) {
            needed = needed.saturating_add(state_object_bytes(&state.layout, &label));
        }
        let mut own: Vec<ColdCacheKey> = plan
            .pages
            .iter()
            .map(|page| page.key)
            .chain(plan.fragment.as_ref().map(|fragment| fragment.key))
            .chain([plan.state_key])
            .collect();
        own.retain(|key| self.keep.insert(*key));
        if !self.make_room(needed, state.modified) {
            for key in own {
                self.keep.remove(&key);
            }
            return Ok(Outcome::Refused);
        }
        let tokens = &state.label.tokens;
        for page in &plan.pages {
            if !self.move_page(tokens, page)? {
                return Ok(Outcome::Unreadable);
            }
        }
        if let Some(fragment) = &plan.fragment
            && !self.move_fragment(tokens, fragment)?
        {
            return Ok(Outcome::Unreadable);
        }
        self.settle()?;
        self.move_state_object(state, plan, &label)
    }

    /// False when a half of the page does not come back from the source.
    fn move_page(&mut self, tokens: &[u32], page: &PagePlan) -> Result<bool, Stop> {
        let tier = self.tier;
        let manager = &tier.manager;
        let present: Vec<ColdCacheKey> = page
            .sources
            .iter()
            .copied()
            .filter(|key| self.blocks.contains_key(key))
            .collect();
        let time = present
            .iter()
            .filter_map(|key| self.blocks.get(key))
            .map(|block| block.modified)
            .max();
        let Some(rows) = page
            .row
            .checked_add(PAGE_BLOCK as usize)
            .and_then(|end| tokens.get(page.row..end))
        else {
            return Err(self.hold());
        };
        if self.on_tier(&page.key) {
            if present.iter().all(|key| self.stays(key)) {
                return Ok(true);
            }
            let read = manager.read_back(page.key, self.fingerprints.kv, &self.page_layout);
            if read.is_none_or(|block| block.tokens != rows) {
                manager.remove(&page.key);
                return Err(Stop);
            }
            #[cfg(test)]
            if cut(hooks::Step::ReadBack(page.key)) {
                return Err(Stop);
            }
            self.raise_time(&page.key, time);
        } else {
            let Some(first) = self.read_source(page.sources[0])? else {
                return Ok(false);
            };
            let Some(second) = self.read_source(page.sources[1])? else {
                return Ok(false);
            };
            let Some(block) = convert_page(page, tokens, &first, &second, &self.fingerprints)
            else {
                return Err(self.hold());
            };
            #[cfg(test)]
            hooks::converted();
            self.write_block(block, time)?;
            self.moved.pages += 1;
        }
        self.read_back.insert(page.key);
        for key in present {
            if !self.stays(&key) && self.free_to_go(&key) {
                self.pending.push((key, ColdGroup::Kv));
            }
        }
        if self.pending.len() >= UNLINK_BATCH {
            self.settle()?;
        }
        Ok(true)
    }

    /// False when the state's tail block does not come back from the
    /// source. The tail block itself stays ([`Self::stays`]).
    fn move_fragment(&mut self, tokens: &[u32], fragment: &FragmentPlan) -> Result<bool, Stop> {
        let tier = self.tier;
        let manager = &tier.manager;
        let time = self
            .blocks
            .get(&fragment.source)
            .map(|block| block.modified);
        let Some(rows) = fragment
            .row
            .checked_add(SOURCE_BLOCK as usize)
            .and_then(|end| tokens.get(fragment.row..end))
        else {
            return Err(self.hold());
        };
        if self.on_tier(&fragment.key) {
            if time.is_none() {
                return Ok(true);
            }
            let read = manager.read_back(
                fragment.key,
                self.fingerprints.kv,
                &self.shape.source_layout,
            );
            if read.is_none_or(|block| block.tokens != rows) {
                manager.remove(&fragment.key);
                return Err(Stop);
            }
            #[cfg(test)]
            if cut(hooks::Step::ReadBack(fragment.key)) {
                return Err(Stop);
            }
            return Ok(true);
        }
        let Some(tail) = self.read_source(fragment.source)? else {
            return Ok(false);
        };
        let Some(block) = convert_fragment(fragment, tokens, &tail, &self.fingerprints) else {
            return Err(self.hold());
        };
        #[cfg(test)]
        hooks::converted();
        self.write_block(block, time)?;
        self.moved.fragments += 1;
        Ok(true)
    }

    /// The state's object, once its whole chain and fragment are in the own
    /// store and their source blocks are gone. An own object of the same
    /// rows that is not this one (another stamp: the own load wrote it
    /// before the old load wrote the source's) gives way to the source's,
    /// once that one is read and converted.
    fn move_state_object(
        &mut self,
        state: &SourceState,
        plan: &StatePlan,
        label: &ColdStateLabel,
    ) -> Result<Outcome, Stop> {
        let tier = self.tier;
        let manager = &tier.manager;
        let key = plan.state_key;
        let fingerprints = self.fingerprints;
        if manager.contains_in(&key, ColdGroup::DFlash2State)
            && manager.state_reads_back(key, fingerprints.state, &state.layout, label, None)
        {
            #[cfg(test)]
            if cut(hooks::Step::ReadBack(key)) {
                return Err(Stop);
            }
            self.raise_time(&key, Some(state.modified));
        } else {
            let mut payload = Vec::new();
            let Some((layout, source_label)) =
                self.reader
                    .read_state(state.key, self.source_state, &mut payload)
            else {
                return Ok(Outcome::Unreadable);
            };
            let Some(object) = convert_state(plan, layout, source_label, payload, &fingerprints)
            else {
                return Ok(Outcome::Unreadable);
            };
            #[cfg(test)]
            hooks::converted();
            manager.remove(&key);
            let (done, landed) = mpsc::channel();
            self.hand_state(object, done)?;
            let Ok(Landed::State {
                key: at,
                ok: true,
                payload,
            }) = landed.recv_timeout(LANDING_WAIT)
            else {
                return Err(Stop);
            };
            if at != key
                || !manager.state_reads_back(
                    key,
                    fingerprints.state,
                    &state.layout,
                    label,
                    Some(&payload),
                )
            {
                manager.remove(&key);
                return Err(Stop);
            }
            #[cfg(test)]
            if cut(hooks::Step::ReadBack(key)) {
                return Err(Stop);
            }
            self.keep_time(&key, Some(state.modified));
        }
        self.pending.push((state.key, ColdGroup::DFlash2State));
        self.settle()?;
        Ok(Outcome::Moved)
    }

    /// The source block for `key`; `None` when it is absent or damaged, as
    /// for the old load. A sound object that is no block of the source's
    /// layout stops the run and holds the source: the source is not what
    /// this load takes it for.
    fn read_source(&mut self, key: ColdCacheKey) -> Result<Option<Box<ColdCacheBlock>>, Stop> {
        match self
            .reader
            .probe_block(key, self.fingerprints.source_kv, &self.shape.source_layout)
        {
            ColdSourceBlock::Read(block) => Ok(Some(block)),
            ColdSourceBlock::Absent | ColdSourceBlock::Damaged => Ok(None),
            ColdSourceBlock::Foreign => Err(self.hold()),
        }
    }

    /// Writes `block` to the own store without eviction, waits for the
    /// writer's word that it landed, reads it back through the restore's
    /// decode and gives it its source's time. Anything else stops the run
    /// with the block's source untouched.
    fn write_block(&mut self, block: ColdCacheBlock, time: Option<u128>) -> Result<(), Stop> {
        let tier = self.tier;
        let manager = &tier.manager;
        let key = block.key;
        let (done, landed) = mpsc::channel();
        if !self.hand_block(block.clone(), done)? {
            self.moved.dropped += 1;
            return Err(Stop);
        }
        if !matches!(
            landed.recv_timeout(LANDING_WAIT),
            Ok(Landed::Block { key: at, ok: true }) if at == key
        ) {
            return Err(Stop);
        }
        if manager
            .read_back(key, self.fingerprints.kv, &block.layout)
            .as_ref()
            != Some(&block)
        {
            manager.remove(&key);
            return Err(Stop);
        }
        #[cfg(test)]
        if cut(hooks::Step::ReadBack(key)) {
            return Err(Stop);
        }
        self.keep_time(&key, time);
        Ok(())
    }

    /// False when the writer's queue took no job: nothing is written.
    fn hand_block(&self, block: ColdCacheBlock, done: mpsc::Sender<Landed>) -> Result<bool, Stop> {
        #[cfg(test)]
        match hooks::at(hooks::Step::Write(block.key)) {
            hooks::Act::Go => {}
            hooks::Act::Cut => return Err(Stop),
            hooks::Act::Drop => return Ok(false),
            hooks::Act::Fail => {
                let _ = done.send(Landed::Block {
                    key: block.key,
                    ok: false,
                });
                return Ok(true);
            }
        }
        self.tier
            .manager
            .enqueue_tracked(block, MakeRoom::Refuse, Instant::now() + LANDING_WAIT, done)
            .map_err(|_| Stop)
    }

    fn hand_state(&self, object: ColdStateObject, done: mpsc::Sender<Landed>) -> Result<(), Stop> {
        #[cfg(test)]
        match hooks::at(hooks::Step::Write(object.key)) {
            hooks::Act::Go => {}
            hooks::Act::Cut | hooks::Act::Drop => return Err(Stop),
            hooks::Act::Fail => {
                let _ = done.send(Landed::State {
                    key: object.key,
                    ok: false,
                    payload: object.payload,
                });
                return Ok(());
            }
        }
        self.tier
            .manager
            .enqueue_state(object, MakeRoom::Refuse, done)
            .map_err(|_| Stop)
    }

    /// The source objects whose rows the own store holds and read back go,
    /// after the device took what the writer handed it: a landed write
    /// alone does not survive a power cut, and these objects are the only
    /// other copy.
    fn settle(&mut self) -> Result<(), Stop> {
        if self.pending.is_empty() {
            return Ok(());
        }
        #[cfg(test)]
        if cut(hooks::Step::Flush) {
            return Err(Stop);
        }
        if !self.tier.manager.flush_device() {
            return Err(Stop);
        }
        for (key, group) in std::mem::take(&mut self.pending) {
            #[cfg(test)]
            if cut(hooks::Step::Unlink(key)) {
                return Err(Stop);
            }
            if !self.reader.remove_moved(key, group) {
                return Err(Stop);
            }
            self.blocks.remove(&key);
        }
        Ok(())
    }

    /// The end of a run that left no state the old load takes back in the
    /// source: every moved state's chain, fragment and object are in the
    /// own store and on the device, then the source directory goes under
    /// its lock, the odd states' tail blocks with it.
    fn remove_source(&self, plans: &[&StatePlan]) -> bool {
        let manager = &self.tier.manager;
        let whole = plans.iter().all(|plan| {
            plan.pages
                .iter()
                .map(|page| &page.key)
                .chain(plan.fragment.as_ref().map(|fragment| &fragment.key))
                .all(|key| manager.contains_in(key, ColdGroup::Kv))
                && manager.contains_in(&plan.state_key, ColdGroup::DFlash2State)
        });
        if !whole || !manager.flush_device() {
            return false;
        }
        #[cfg(test)]
        if cut(hooks::Step::Remove) {
            return false;
        }
        self.tier.remove_source()
    }
}

#[cfg(test)]
pub(crate) mod hooks {
    use std::cell::{Cell, RefCell};

    use mlx_paged_attn::ColdCacheKey;

    /// A point of a run a test watches: a write about to be handed to the
    /// writer, an object that read back, a device flush about to be asked
    /// for, a source object about to go, the source directory about to go.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Step {
        Write(ColdCacheKey),
        ReadBack(ColdCacheKey),
        Flush,
        Unlink(ColdCacheKey),
        Remove,
    }

    /// What the run does at a step: go on; end as a killed process does;
    /// at a write, find the queue full or have the write fail.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Act {
        Go,
        Cut,
        Drop,
        Fail,
    }

    type Observer = Box<dyn FnMut(Step) -> Act>;

    thread_local! {
        static OBSERVER: RefCell<Option<Observer>> = const { RefCell::new(None) };
        static CONVERSIONS: Cell<u64> = const { Cell::new(0) };
    }

    pub(crate) fn observe(observer: Option<Observer>) {
        OBSERVER.with(|slot| *slot.borrow_mut() = observer);
    }

    pub(super) fn at(step: Step) -> Act {
        OBSERVER.with(|slot| {
            slot.borrow_mut()
                .as_mut()
                .map_or(Act::Go, |observer| observer(step))
        })
    }

    pub(super) fn converted() {
        CONVERSIONS.with(|count| count.set(count.get() + 1));
    }

    /// The converter calls this thread's runs made; reset by the read.
    pub(crate) fn take_conversions() -> u64 {
        CONVERSIONS.with(Cell::take)
    }
}
