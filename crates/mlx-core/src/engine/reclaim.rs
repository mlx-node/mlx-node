//! One take order for a refused KV reservation, after Splash
//! `MemoryLedger::free`: the units that cost nothing, a need met in place
//! takes nothing, a count that cannot cover the need takes nothing, else
//! units go oldest first until the need is met, then the rebuilt caches
//! nothing holds, then, for a need that reaches them, the states in use
//! (Splash `Order::Stage` Rebuild before InUse).

use std::collections::HashSet;

use crate::models::qwen3_5::dflash2_state::store::StoreStep;
use crate::transformer::paged_kv_cache_adapter::SeqId;

/// Blocks a refused attempt takes in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvNeed {
    pub blocks: u32,
    pub over_takeable: bool,
    /// The caller ran the met check just now: its grow and room stand.
    pub checked: bool,
    /// The lone running turn's need reaches the states turns use, after
    /// every ordinary unit (Splash `Engine::growthClass`,
    /// `ReclaimClass::InUse`).
    pub reach_in_use: bool,
    /// Splash `TakeRights::rebuiltPools`: a `Growth` claim, or one ranked
    /// below a running lane, takes no rebuilt cache.
    pub rebuilt: bool,
}

impl KvNeed {
    pub(crate) fn in_place(blocks: u32) -> Self {
        Self {
            blocks,
            over_takeable: false,
            checked: false,
            reach_in_use: false,
            rebuilt: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum UnitKey {
    Store(StoreStep),
    Ladder {
        owner_id: String,
        prefix_len: u32,
        final_block_hash: u64,
    },
    History,
    Owner(SeqId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Unit {
    pub stamp: u64,
    pub key: UnitKey,
}

/// Upper bounds: blocks the units make takeable, bytes they free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reach {
    pub blocks: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Freed {
    /// The need was met in place; nothing was taken.
    Met,
    /// Units were taken (the need may still be short: retry).
    Took,
    /// Nothing was taken.
    Short,
}

/// Splash `Engine::allocate`'s `met` list for one retry loop: a need is
/// recorded after it came back Met or after a take, and the same need
/// again stops the loop.
#[derive(Debug)]
pub(crate) struct MetBefore<K: PartialEq> {
    met: Vec<K>,
}

impl<K: PartialEq> Default for MetBefore<K> {
    fn default() -> Self {
        Self { met: Vec::new() }
    }
}

impl<K: PartialEq> MetBefore<K> {
    /// Whether `need` was met or freed for once already (stop).
    pub(crate) fn seen(&self, need: &K) -> bool {
        self.met.contains(need)
    }

    /// Whether the attempt runs again after `freed`; records the need.
    pub(crate) fn retry(&mut self, need: K, freed: Freed) -> bool {
        if freed == Freed::Short {
            return false;
        }
        self.met.push(need);
        true
    }

    pub(crate) fn clear(&mut self) {
        self.met.clear();
    }
}

pub(crate) trait KvReclaim {
    fn at_cap(&self) -> bool;
    fn takeable(&self) -> u32;
    fn ungrown(&self) -> u32;
    fn bytes_per_block(&self) -> u64;
    fn met(&mut self, need: KvNeed) -> bool;
    fn grow_room(&self) -> u32;
    fn sync_marks(&mut self) {}
    /// What the units of the need's classes free; may over-count.
    fn reach(&self, at_cap: bool, in_use: bool) -> Reach;
    fn compact_one(&mut self) -> bool;
    /// The oldest state unit of one class: in use by a turn, or not.
    fn oldest_state(&self, at_cap: bool, skip: &HashSet<UnitKey>, in_use: bool) -> Option<Unit>;
    fn oldest_kv(&self, _at_cap: bool, _skip: &HashSet<UnitKey>) -> Option<Unit> {
        None
    }
    /// Takes the unit; false when it freed nothing.
    fn take(&mut self, key: &UnitKey) -> bool;
    /// Bytes of the rebuilt caches nothing holds.
    fn rebuilt_bytes(&self) -> u64 {
        0
    }
    /// Drops one rebuilt-cache unit; false when none is left.
    fn rebuild_one(&mut self) -> bool {
        false
    }
}

/// A state goes first on a tie, as Splash `Cache::nextUnit` orders them.
pub(crate) fn older_first(state: Option<Unit>, kv: Option<Unit>) -> Option<Unit> {
    match (state, kv) {
        (Some(state), Some(kv)) => Some(if state.stamp <= kv.stamp { state } else { kv }),
        (state, kv) => state.or(kv),
    }
}

/// Splash `MemoryLedger::free` for a KV need. Marks are synced first: a
/// stale mark would keep a chain out of both the met check and the count.
/// A unit that frees nothing is passed over (`MemoryLedger::take`); a grow
/// in a met miss can reach the cap, so the cap is read again after each.
/// Every reach term may over-count, never under-count: an under-count
/// fails a turn a take would save. A need that reaches in-use states takes
/// them only once no ordinary unit is left, oldest first (Splash
/// `Cache::nextUnit`'s `Phase::InUse`).
pub(crate) fn free_for_kv<U: KvReclaim>(units: &mut U, need: KvNeed) -> Freed {
    crate::array::clear_cache();
    units.sync_marks();
    if !need.checked && units.met(need) {
        return Freed::Met;
    }
    let mut at_cap = units.at_cap();
    let reach = units.reach(at_cap, need.reach_in_use);
    // A page need is met by pages only: a rebuilt cache gives bytes.
    let reach_bytes = if need.rebuilt && !at_cap {
        reach.bytes.saturating_add(units.rebuilt_bytes())
    } else {
        reach.bytes
    };
    let grow = if at_cap {
        0
    } else {
        let from_bytes = reach_bytes.div_ceil(units.bytes_per_block().max(1));
        u64::from(units.ungrown()).min(u64::from(units.grow_room()).saturating_add(from_bytes))
    };
    let most = u64::from(units.takeable())
        .saturating_add(reach.blocks)
        .saturating_add(grow);
    if most < u64::from(need.blocks) {
        return Freed::Short;
    }
    let mut skip = HashSet::new();
    let mut took = false;
    let mut met = false;
    loop {
        let freed = if !at_cap && units.compact_one() {
            true
        } else {
            let state = units.oldest_state(at_cap, &skip, false);
            let kv = units.oldest_kv(at_cap, &skip);
            match older_first(state, kv) {
                Some(unit) => take_one(units, unit.key, &mut skip),
                None if need.rebuilt && !at_cap && units.rebuild_one() => true,
                None => break,
            }
        };
        if freed {
            took = true;
            if units.met(need) {
                met = true;
                break;
            }
            at_cap = units.at_cap();
        }
    }
    while need.reach_in_use && !met {
        let Some(unit) = units.oldest_state(at_cap, &skip, true) else {
            break;
        };
        if take_one(units, unit.key, &mut skip) {
            took = true;
            met = units.met(need);
            at_cap = units.at_cap();
        }
    }
    if took {
        crate::array::clear_cache();
        Freed::Took
    } else {
        Freed::Short
    }
}

/// Takes `key`; one that frees nothing is passed over from then on.
fn take_one<U: KvReclaim>(units: &mut U, key: UnitKey, skip: &mut HashSet<UnitKey>) -> bool {
    let freed = units.take(&key);
    if !freed {
        units.sync_marks();
        skip.insert(key);
    }
    freed
}

/// A refused draft fit: the bytes it lacked at `queries` verifier rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DraftRefusal {
    pub bytes: u64,
    pub queries: u32,
}

/// The units a draft need takes, by their own stamps.
pub(crate) trait DraftReclaim {
    /// Bytes the refused draft fits lack now; 0 when they fit.
    fn short(&self) -> u64;
    /// Bytes every unit could give the fits (an upper bound).
    fn reach(&self) -> u64;
    fn oldest(&self, skip: &HashSet<UnitKey>) -> Option<Unit>;
    /// Takes the unit; false when it freed nothing.
    fn take(&mut self, key: &UnitKey) -> bool;
}

/// Splash `MemoryLedger::free` for a lane's state growth: a need met in
/// place takes nothing, a count that cannot cover it takes nothing, else
/// units go oldest first until it is met; a unit that frees nothing is
/// passed over.
pub(crate) fn free_for_draft<U: DraftReclaim>(units: &mut U) -> Freed {
    crate::array::clear_cache();
    let short = units.short();
    if short == 0 {
        return Freed::Met;
    }
    if units.reach() < short {
        return Freed::Short;
    }
    let mut skip = HashSet::new();
    let mut took = false;
    while let Some(unit) = units.oldest(&skip) {
        if units.take(&unit.key) {
            took = true;
            if units.short() == 0 {
                break;
            }
        } else {
            skip.insert(unit.key);
        }
    }
    if took {
        crate::array::clear_cache();
        Freed::Took
    } else {
        Freed::Short
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Units keyed `Owner(id)`: `(id, stamp, blocks freed)`. A unit whose
    /// take frees nothing stays, as an owner whose release fails does.
    /// Rebuilt units are `(id, bytes)`, oldest first; a take of either kind
    /// logs its id in `taken`. `reach_blocks` adds to the count's blocks.
    #[derive(Default)]
    struct Fake {
        takeable: u32,
        states: Vec<(SeqId, u64, u32)>,
        taken: Vec<SeqId>,
        below_cap: bool,
        ungrown: u32,
        reach_blocks: u64,
        rebuilt: Vec<(SeqId, u64)>,
    }

    impl KvReclaim for Fake {
        fn at_cap(&self) -> bool {
            !self.below_cap
        }
        fn takeable(&self) -> u32 {
            self.takeable
        }
        fn ungrown(&self) -> u32 {
            self.ungrown
        }
        fn bytes_per_block(&self) -> u64 {
            1
        }
        fn met(&mut self, need: KvNeed) -> bool {
            self.takeable >= need.blocks
        }
        fn grow_room(&self) -> u32 {
            0
        }
        fn reach(&self, _at_cap: bool, _in_use: bool) -> Reach {
            Reach {
                blocks: self
                    .states
                    .iter()
                    .map(|&(_, _, frees)| u64::from(frees))
                    .sum::<u64>()
                    .saturating_add(self.reach_blocks),
                bytes: 0,
            }
        }
        fn compact_one(&mut self) -> bool {
            false
        }
        fn oldest_state(
            &self,
            _at_cap: bool,
            skip: &HashSet<UnitKey>,
            in_use: bool,
        ) -> Option<Unit> {
            if in_use {
                return None;
            }
            self.states
                .iter()
                .filter(|&&(id, _, _)| !skip.contains(&UnitKey::Owner(id)))
                .min_by_key(|&&(_, stamp, _)| stamp)
                .map(|&(id, stamp, _)| Unit {
                    stamp,
                    key: UnitKey::Owner(id),
                })
        }
        fn take(&mut self, key: &UnitKey) -> bool {
            let UnitKey::Owner(id) = key else {
                return false;
            };
            self.taken.push(*id);
            let Some(index) = self.states.iter().position(|&(unit, _, _)| unit == *id) else {
                return false;
            };
            let frees = self.states[index].2;
            if frees == 0 {
                return false;
            }
            self.states.remove(index);
            self.takeable += frees;
            true
        }
        fn rebuilt_bytes(&self) -> u64 {
            self.rebuilt.iter().map(|&(_, bytes)| bytes).sum()
        }
        fn rebuild_one(&mut self) -> bool {
            if self.rebuilt.is_empty() {
                return false;
            }
            let (id, _) = self.rebuilt.remove(0);
            self.taken.push(id);
            self.takeable += 1;
            true
        }
    }

    /// Owners `(id, stamp, bytes freed)` against a `short` byte need.
    #[derive(Default)]
    struct DraftFake {
        short: u64,
        owners: Vec<(SeqId, u64, u64)>,
        taken: Vec<SeqId>,
    }

    impl DraftReclaim for DraftFake {
        fn short(&self) -> u64 {
            self.short
        }
        fn reach(&self) -> u64 {
            self.owners.iter().map(|&(_, _, frees)| frees).sum()
        }
        fn oldest(&self, skip: &HashSet<UnitKey>) -> Option<Unit> {
            self.owners
                .iter()
                .filter(|&&(id, _, _)| !skip.contains(&UnitKey::Owner(id)))
                .min_by_key(|&&(_, stamp, _)| stamp)
                .map(|&(id, stamp, _)| Unit {
                    stamp,
                    key: UnitKey::Owner(id),
                })
        }
        fn take(&mut self, key: &UnitKey) -> bool {
            let UnitKey::Owner(id) = key else {
                return false;
            };
            self.taken.push(*id);
            let Some(index) = self.owners.iter().position(|&(unit, _, _)| unit == *id) else {
                return false;
            };
            let frees = self.owners[index].2;
            if frees == 0 {
                return false;
            }
            self.owners.remove(index);
            self.short = self.short.saturating_sub(frees);
            true
        }
    }

    fn clear_cache_calls() -> u32 {
        crate::array::memory::TEST_CLEAR_CACHE_CALLS.with(std::cell::Cell::get)
    }

    /// Mutation: `<` for `<=` in `older_first` → red (the KV unit wins).
    #[test]
    fn older_first_gives_ties_to_the_state() {
        let unit = |stamp, seq_id| Unit {
            stamp,
            key: UnitKey::Owner(seq_id),
        };
        assert_eq!(
            older_first(Some(unit(5, 1)), Some(unit(5, 2))),
            Some(unit(5, 1))
        );
        assert_eq!(
            older_first(Some(unit(6, 1)), Some(unit(5, 2))),
            Some(unit(5, 2))
        );
        assert_eq!(older_first(None, Some(unit(5, 2))), Some(unit(5, 2)));
    }

    /// Mutation: `if !freed { break }` in `free_for_kv` → red (`Short`).
    #[test]
    fn a_unit_that_frees_nothing_is_passed_over() {
        let mut fake = Fake {
            states: vec![(1, 1, 0), (2, 2, 2)],
            ..Fake::default()
        };
        assert_eq!(free_for_kv(&mut fake, KvNeed::in_place(2)), Freed::Took);
        assert_eq!(fake.taken, vec![1, 2]);
        assert_eq!(fake.states, vec![(1, 1, 0)], "A stays, passed over");
    }

    /// Mutation: clear the cache after every taken unit in `free_for_kv` →
    /// red (+4 ≠ +2).
    #[test]
    fn clear_cache_runs_before_the_count_and_once_after_the_take() {
        let mut fake = Fake {
            states: vec![(1, 1, 1), (2, 2, 1), (3, 3, 1)],
            ..Fake::default()
        };
        let before = clear_cache_calls();
        assert_eq!(free_for_kv(&mut fake, KvNeed::in_place(3)), Freed::Took);
        assert_eq!(fake.taken, vec![1, 2, 3]);
        assert_eq!(clear_cache_calls() - before, 2);

        let mut short = Fake {
            states: vec![(1, 1, 1)],
            ..Fake::default()
        };
        let before = clear_cache_calls();
        assert_eq!(free_for_kv(&mut short, KvNeed::in_place(3)), Freed::Short);
        assert!(short.taken.is_empty());
        assert_eq!(clear_cache_calls() - before, 1);
    }

    /// Below the cap: two store steps that free bytes and an owner that
    /// frees a block. The met miss after the first take grows to the cap.
    struct CapFake {
        at_cap: bool,
        takeable: u32,
        stores: Vec<u64>,
        owner: Option<u64>,
        taken: Vec<UnitKey>,
    }

    fn store_key(entry: u64) -> UnitKey {
        UnitKey::Store(StoreStep {
            entry,
            kind: crate::models::qwen3_5::dflash2_state::store::StoreStepKind::DropEnd,
        })
    }

    impl KvReclaim for CapFake {
        fn at_cap(&self) -> bool {
            self.at_cap
        }
        fn takeable(&self) -> u32 {
            self.takeable
        }
        fn ungrown(&self) -> u32 {
            if self.at_cap { 0 } else { 4 }
        }
        fn bytes_per_block(&self) -> u64 {
            1
        }
        fn met(&mut self, need: KvNeed) -> bool {
            if !self.taken.is_empty() {
                self.at_cap = true;
            }
            self.takeable >= need.blocks
        }
        fn grow_room(&self) -> u32 {
            0
        }
        fn reach(&self, at_cap: bool, _in_use: bool) -> Reach {
            Reach {
                blocks: u64::from(self.owner.is_some()),
                bytes: if at_cap { 0 } else { self.stores.len() as u64 },
            }
        }
        fn compact_one(&mut self) -> bool {
            false
        }
        fn oldest_state(
            &self,
            at_cap: bool,
            skip: &HashSet<UnitKey>,
            in_use: bool,
        ) -> Option<Unit> {
            if at_cap || in_use {
                return None;
            }
            self.stores
                .iter()
                .map(|&entry| Unit {
                    stamp: entry,
                    key: store_key(entry),
                })
                .find(|unit| !skip.contains(&unit.key))
        }
        fn oldest_kv(&self, _at_cap: bool, skip: &HashSet<UnitKey>) -> Option<Unit> {
            self.owner
                .map(|stamp| Unit {
                    stamp,
                    key: UnitKey::Owner(9),
                })
                .filter(|unit| !skip.contains(&unit.key))
        }
        fn take(&mut self, key: &UnitKey) -> bool {
            self.taken.push(key.clone());
            match key {
                UnitKey::Store(step) => {
                    self.stores.retain(|&entry| entry != step.entry);
                    true
                }
                UnitKey::Owner(_) => {
                    self.owner = None;
                    self.takeable += 1;
                    true
                }
                _ => false,
            }
        }
    }

    /// A grow inside a met miss reaches the cap: the newer store step
    /// frees no block there and stays.
    /// Mutation: read `at_cap` once in `free_for_kv` → red (store 2 taken).
    #[test]
    fn walk_rereads_the_cap_after_a_grow() {
        let mut fake = CapFake {
            at_cap: false,
            takeable: 0,
            stores: vec![1, 2],
            owner: Some(3),
            taken: Vec::new(),
        };
        assert_eq!(free_for_kv(&mut fake, KvNeed::in_place(1)), Freed::Took);
        assert_eq!(fake.taken, vec![store_key(1), UnitKey::Owner(9)]);
        assert_eq!(fake.stores, vec![2], "store 2 frees no block at the cap");
    }

    /// At the cap: ordinary units and units a turn uses, `(id, stamp)`,
    /// each freeing one block.
    #[derive(Default)]
    struct UseFake {
        takeable: u32,
        ordinary: Vec<(SeqId, u64)>,
        used: Vec<(SeqId, u64)>,
        taken: Vec<SeqId>,
    }

    impl KvReclaim for UseFake {
        fn at_cap(&self) -> bool {
            true
        }
        fn takeable(&self) -> u32 {
            self.takeable
        }
        fn ungrown(&self) -> u32 {
            0
        }
        fn bytes_per_block(&self) -> u64 {
            1
        }
        fn met(&mut self, need: KvNeed) -> bool {
            self.takeable >= need.blocks
        }
        fn grow_room(&self) -> u32 {
            0
        }
        fn reach(&self, _at_cap: bool, in_use: bool) -> Reach {
            let used = if in_use { self.used.len() } else { 0 };
            Reach {
                blocks: (self.ordinary.len() + used) as u64,
                bytes: 0,
            }
        }
        fn compact_one(&mut self) -> bool {
            false
        }
        fn oldest_state(
            &self,
            _at_cap: bool,
            skip: &HashSet<UnitKey>,
            in_use: bool,
        ) -> Option<Unit> {
            if in_use { &self.used } else { &self.ordinary }
                .iter()
                .map(|&(id, stamp)| Unit {
                    stamp,
                    key: UnitKey::Owner(id),
                })
                .filter(|unit| !skip.contains(&unit.key))
                .min_by_key(|unit| unit.stamp)
        }
        fn take(&mut self, key: &UnitKey) -> bool {
            let UnitKey::Owner(id) = key else {
                return false;
            };
            self.ordinary.retain(|&(unit, _)| unit != *id);
            self.used.retain(|&(unit, _)| unit != *id);
            self.taken.push(*id);
            self.takeable += 1;
            true
        }
    }

    /// A need that reaches in-use units counts them and takes every ordinary
    /// unit first, though the in-use one is older; one that does not reach
    /// them neither counts nor takes them.
    /// Mutations: take in-use units in the first loop by stamp beside the
    /// ordinary ones → red ([1] ≠ [2]); count with `reach_in_use` false in
    /// `free_for_kv` → red (the 3-block reach is `Short`).
    #[test]
    fn in_use_units_go_after_every_ordinary_unit() {
        let fake = || UseFake {
            ordinary: vec![(2, 10), (3, 20)],
            used: vec![(1, 0)],
            ..UseFake::default()
        };
        let reach = |blocks| KvNeed {
            reach_in_use: true,
            ..KvNeed::in_place(blocks)
        };
        let mut one = fake();
        assert_eq!(free_for_kv(&mut one, reach(1)), Freed::Took);
        assert_eq!(one.taken, vec![2]);
        let mut all = fake();
        assert_eq!(free_for_kv(&mut all, reach(3)), Freed::Took);
        assert_eq!(all.taken, vec![2, 3, 1]);
        let mut ordinary = fake();
        assert_eq!(free_for_kv(&mut ordinary, KvNeed::in_place(2)), Freed::Took);
        assert_eq!(ordinary.taken, vec![2, 3], "the used unit stays");
        let mut short = fake();
        assert_eq!(free_for_kv(&mut short, KvNeed::in_place(3)), Freed::Short);
        assert!(short.taken.is_empty());
    }

    /// Mutation: record the need in `MetBefore::retry` only on `Met` → red
    /// (`seen(3)` stays false after a take).
    #[test]
    fn a_repeated_need_after_a_take_stops() {
        let mut met = MetBefore::<u32>::default();
        assert!(!met.seen(&3));
        assert!(met.retry(3, Freed::Took));
        assert!(met.seen(&3));
        assert!(!met.retry(2, Freed::Short));
        assert!(!met.seen(&2));
        assert!(met.retry(2, Freed::Met));
        assert!(met.seen(&2));
    }

    /// At the cap the need is pages only: the rebuilt unit is no unit and
    /// its bytes are out of the count.
    /// Mutation: drop `!at_cap` from the `rebuild_one` arm in `free_for_kv`
    /// → red (`Took`, the rebuilt unit taken).
    #[test]
    fn no_rebuilt_unit_at_the_cap() {
        let mut fake = Fake {
            states: vec![(1, 1, 0)],
            reach_blocks: 1,
            rebuilt: vec![(7, 1)],
            ..Fake::default()
        };
        assert_eq!(free_for_kv(&mut fake, KvNeed::in_place(1)), Freed::Short);
        assert_eq!(fake.taken, vec![1], "only the state was tried");
        assert_eq!(fake.rebuilt, vec![(7, 1)]);
    }

    /// Mutation: ignore `need.rebuilt` in `free_for_kv` → red (the claim
    /// without the right takes the rebuilt unit).
    #[test]
    fn a_claim_without_the_rebuild_right_takes_no_rebuilt_unit() {
        let fake = || Fake {
            below_cap: true,
            ungrown: 4,
            rebuilt: vec![(7, 1)],
            ..Fake::default()
        };
        let mut without = fake();
        let need = KvNeed {
            rebuilt: false,
            ..KvNeed::in_place(1)
        };
        assert_eq!(free_for_kv(&mut without, need), Freed::Short);
        assert!(without.taken.is_empty());
        let mut with = fake();
        assert_eq!(free_for_kv(&mut with, KvNeed::in_place(1)), Freed::Took);
        assert_eq!(with.taken, vec![7]);
    }

    /// The count covers the need (state 1 + grow 1 from the rebuilt bytes):
    /// the state goes first, then the rebuilt unit.
    /// Mutation: try `rebuild_one` before `older_first` in `free_for_kv` →
    /// red (`[7, 1]`).
    #[test]
    fn rebuilt_units_go_after_every_free_now_unit() {
        let mut fake = Fake {
            below_cap: true,
            ungrown: 1,
            states: vec![(1, 9, 1)],
            rebuilt: vec![(7, 1)],
            ..Fake::default()
        };
        assert_eq!(free_for_kv(&mut fake, KvNeed::in_place(2)), Freed::Took);
        assert_eq!(fake.taken, vec![1, 7]);
    }

    /// Mutation: drop `rebuilt_bytes` from `reach_bytes` in `free_for_kv` →
    /// red (`Short`: grow 0).
    #[test]
    fn the_count_holds_rebuilt_bytes_below_the_cap() {
        let mut fake = Fake {
            below_cap: true,
            ungrown: 4,
            rebuilt: vec![(7, 1)],
            ..Fake::default()
        };
        assert_eq!(free_for_kv(&mut fake, KvNeed::in_place(1)), Freed::Took);
        assert_eq!(fake.taken, vec![7]);
    }

    /// Mutation: `break` when `take` frees nothing in `free_for_draft` →
    /// red (`Short`).
    #[test]
    fn free_for_draft_passes_over_an_owner_that_frees_nothing() {
        let mut fake = DraftFake {
            short: 5,
            owners: vec![(1, 1, 0), (2, 2, 5)],
            ..DraftFake::default()
        };
        assert_eq!(free_for_draft(&mut fake), Freed::Took);
        assert_eq!(fake.taken, vec![1, 2]);
        assert_eq!(fake.owners, vec![(1, 1, 0)], "A stays, passed over");
        assert_eq!(fake.short, 0);
    }

    /// Mutation: drop the reach check in `free_for_draft` → red (the owner
    /// is taken).
    #[test]
    fn a_doomed_draft_need_takes_nothing() {
        let mut fake = DraftFake {
            short: 10,
            owners: vec![(1, 1, 9)],
            ..DraftFake::default()
        };
        let before = clear_cache_calls();
        assert_eq!(free_for_draft(&mut fake), Freed::Short);
        assert!(fake.taken.is_empty());
        assert_eq!(clear_cache_calls() - before, 1);
    }
}
