//! One take order for a refused KV reservation, after Splash
//! `MemoryLedger::free`: the units that cost nothing, a need met in place
//! takes nothing, a count that cannot cover the need takes nothing, else
//! units go oldest first until the need is met.

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
}

impl KvNeed {
    pub(crate) fn in_place(blocks: u32) -> Self {
        Self {
            blocks,
            over_takeable: false,
            checked: false,
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
    Table(SeqId),
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
    fn before_count(&mut self, _at_cap: bool) {}
    fn sync_marks(&mut self) {}
    fn reach(&self, at_cap: bool) -> Reach;
    fn compact_one(&mut self) -> bool;
    fn oldest_state(&self, at_cap: bool, skip: &HashSet<UnitKey>) -> Option<Unit>;
    fn oldest_kv(&self, _at_cap: bool, _skip: &HashSet<UnitKey>) -> Option<Unit> {
        None
    }
    /// Takes the unit; false when it freed nothing.
    fn take(&mut self, key: &UnitKey) -> bool;
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
/// A unit that frees nothing is passed over (`MemoryLedger::take`). Every reach term may over-count,
/// never under-count: an under-count fails a turn a take would save.
pub(crate) fn free_for_kv<U: KvReclaim>(units: &mut U, need: KvNeed) -> Freed {
    crate::array::clear_cache();
    units.sync_marks();
    if !need.checked && units.met(need) {
        return Freed::Met;
    }
    let at_cap = units.at_cap();
    units.before_count(at_cap);
    let reach = units.reach(at_cap);
    let grow = if at_cap {
        0
    } else {
        let from_bytes = reach.bytes.div_ceil(units.bytes_per_block().max(1));
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
    loop {
        let freed = if !at_cap && units.compact_one() {
            true
        } else {
            let state = units.oldest_state(at_cap, &skip);
            let kv = units.oldest_kv(at_cap, &skip);
            let Some(unit) = older_first(state, kv) else {
                break;
            };
            let freed = units.take(&unit.key);
            if !freed {
                units.sync_marks();
                skip.insert(unit.key);
            }
            freed
        };
        if freed {
            took = true;
            if units.met(need) {
                break;
            }
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
    #[derive(Default)]
    struct Fake {
        takeable: u32,
        states: Vec<(SeqId, u64, u32)>,
        taken: Vec<SeqId>,
    }

    impl KvReclaim for Fake {
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
        fn reach(&self, _at_cap: bool) -> Reach {
            Reach {
                blocks: self
                    .states
                    .iter()
                    .map(|&(_, _, frees)| u64::from(frees))
                    .sum(),
                bytes: 0,
            }
        }
        fn compact_one(&mut self) -> bool {
            false
        }
        fn oldest_state(&self, _at_cap: bool, skip: &HashSet<UnitKey>) -> Option<Unit> {
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
}
