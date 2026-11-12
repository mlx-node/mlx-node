//! Content-keyed resume states for DFlash2 turns (Splash `StateCache`).
//!
//! An entry is one state on one complete KV block: the GDN cell and draft
//! window at a block-aligned row of a token lineage. It holds no target KV:
//! the KV is the prefix cache's chain, which the entry keeps resident by
//! marking the block it sits on (Splash `StateMarks`). Lookup keys on the
//! paged allocator's chained block hashes plus an exact token compare and the
//! cache salt; the owner id plays no part. Memory alone bounds the store: the
//! governor takes its entries as units (Splash has no count or byte cap).
//!
//! Every byte the store holds is an MLX array the allocator already counts, so
//! nothing here registers with `cache_limit::coordinator()`.

use std::collections::HashMap;
use std::sync::Arc;

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::*;

use crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes;

/// Bytes a parked state keeps alive.
pub(crate) trait StateBytes {
    fn state_bytes(&self) -> u64;
}

/// The prompt or history a key is computed from. Blocks past the end of
/// `extra_keys` carry no extra keys.
#[derive(Clone, Copy)]
pub(crate) struct Lineage<'a> {
    pub(crate) tokens: &'a [u32],
    pub(crate) extra_keys: &'a [Vec<u64>],
    pub(crate) salt: u64,
}

impl<'a> Lineage<'a> {
    pub(crate) fn text(tokens: &'a [u32], salt: u64) -> Self {
        Self {
            tokens,
            extra_keys: &[],
            salt,
        }
    }
}

pub(crate) type EntryId = u64;

/// A publication's class (Splash `StateClass` as `StateCache` keeps it): a
/// checkpoint is disposable and goes before ordinary states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateClass {
    Checkpoint,
    Ordinary,
}

/// What [`HybridStateStore::insert`] did: `new` is false when a state on the
/// same block was reused, `upgraded` when that reuse made a checkpoint
/// ordinary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Inserted {
    pub(crate) id: EntryId,
    pub(crate) new: bool,
    pub(crate) upgraded: bool,
}

pub(crate) struct Entry<S> {
    id: EntryId,
    block_size: u32,
    salt: u64,
    tokens: Vec<u32>,
    chain: Vec<u64>,
    state: S,
    state_bytes: u64,
    last_used: u64,
    checkpoint: bool,
    publication: u64,
}

impl<S> Entry<S> {
    pub(crate) fn id(&self) -> EntryId {
        self.id
    }

    /// The row the state resumes at: its lineage's length.
    pub(crate) fn depth(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Chain hashes of the entry's whole blocks.
    pub(crate) fn chain(&self) -> &[u64] {
        &self.chain
    }

    /// The block the state sits on (Splash keys a state by its KV block).
    pub(crate) fn mark(&self) -> u64 {
        self.chain.last().copied().unwrap_or_default()
    }

    pub(crate) fn state(&self) -> &S {
        &self.state
    }

    /// The state, its lineage's token and chain vectors included.
    pub(crate) fn state_bytes(&self) -> u64 {
        self.state_bytes
    }

    pub(crate) fn last_used(&self) -> u64 {
        self.last_used
    }

    pub(crate) fn checkpoint(&self) -> bool {
        self.checkpoint
    }

    /// Set once when the entry is made; an upgrade keeps it (Splash
    /// `StateCache::entryFor`, `makeOrdinary`).
    pub(crate) fn publication(&self) -> u64 {
        self.publication
    }

    /// The entry's lineage equals `prompt[..depth]`, `depth < |prompt|`.
    fn resumes(&self, prompt: &PromptKey<'_>) -> bool {
        let depth = self.depth();
        let blocks = self.chain.len();
        depth > 0
            && depth < prompt.tokens.len()
            && self.block_size == prompt.block_size
            && self.salt == prompt.salt
            && prompt.chain.get(..blocks) == Some(&self.chain[..])
            && prompt.tokens[..depth] == self.tokens[..]
    }
}

struct PromptKey<'a> {
    tokens: &'a [u32],
    chain: Vec<u64>,
    block_size: u32,
    salt: u64,
}

fn block_chain(lineage: Lineage<'_>, block_size: u32) -> Vec<u64> {
    let bs = block_size as usize;
    let blocks = lineage.tokens.len() / bs;
    let Ok(prefix_len) = u32::try_from(blocks * bs) else {
        return Vec::new();
    };
    if blocks == 0 {
        return Vec::new();
    }
    let extra_keys: Vec<Vec<u64>> = (0..blocks)
        .map(|block| lineage.extra_keys.get(block).cloned().unwrap_or_default())
        .collect();
    compute_paged_prefix_block_hashes(
        lineage.tokens,
        prefix_len,
        block_size,
        &extra_keys,
        lineage.salt,
    )
    .unwrap_or_default()
}

/// The bytes an entry's lineage vectors hold.
pub(crate) fn lineage_bytes(tokens: usize, chain: usize) -> u64 {
    (tokens as u64)
        .saturating_mul(std::mem::size_of::<u32>() as u64)
        .saturating_add((chain as u64).saturating_mul(std::mem::size_of::<u64>() as u64))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StoreHit {
    pub(crate) id: EntryId,
    pub(crate) depth: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StoreBytes {
    pub(crate) entries: usize,
    pub(crate) state_bytes: u64,
}

pub(crate) struct HybridStateStore<S> {
    block_size: u32,
    entries: Vec<Entry<S>>,
    next_id: EntryId,
    publications: u64,
    /// Blocks an unfinished turn resumes from, with an entry there or not
    /// (Splash `StateCache::uses_`).
    uses: HashMap<u64, u32>,
    recency: Arc<CacheRecency>,
}

impl<S: StateBytes> HybridStateStore<S> {
    pub(crate) fn new(block_size: u32) -> Self {
        Self {
            block_size: block_size.max(1),
            entries: Vec::new(),
            next_id: 1,
            publications: 0,
            uses: HashMap::new(),
            recency: Arc::new(CacheRecency::new()),
        }
    }

    /// Stamp from `recency` from now on, the clock the model's other caches
    /// share.
    pub(crate) fn share_recency(&mut self, recency: Arc<CacheRecency>) {
        self.recency = recency;
    }

    #[cfg(test)]
    pub(crate) fn block_size(&self) -> u32 {
        self.block_size
    }

    pub(crate) fn bytes(&self) -> StoreBytes {
        StoreBytes {
            entries: self.entries.len(),
            state_bytes: self
                .entries
                .iter()
                .fold(0, |sum, entry| sum.saturating_add(entry.state_bytes)),
        }
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = &Entry<S>> {
        self.entries.iter()
    }

    pub(crate) fn get(&self, id: EntryId) -> Option<&Entry<S>> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    fn get_mut(&mut self, id: EntryId) -> Option<&mut Entry<S>> {
        self.entries.iter_mut().find(|entry| entry.id == id)
    }

    fn prompt_key<'a>(&self, lineage: Lineage<'a>) -> PromptKey<'a> {
        PromptKey {
            tokens: lineage.tokens,
            chain: block_chain(lineage, self.block_size),
            block_size: self.block_size,
            salt: lineage.salt,
        }
    }

    /// Publishes `state` at the end of `lineage`, which must be a whole
    /// number of blocks (Splash `Cache::blockAt`). A state already on the
    /// same block of the same lineage is reused and restamped instead, an
    /// ordinary publication upgrading a checkpoint there (Splash
    /// `reuseCompositeState`: one state per block). A checkpoint at a block
    /// in use is ordinary at once (Splash `StateCache::publicationEntry`).
    pub(crate) fn insert(
        &mut self,
        lineage: Lineage<'_>,
        state: S,
        class: StateClass,
    ) -> Result<Inserted> {
        let row = lineage.tokens.len();
        let bs = self.block_size as usize;
        if row == 0 || !row.is_multiple_of(bs) {
            return Err(Error::from_reason(format!(
                "DFlash2 state at row {row} is not a complete KV block (block size {bs})"
            )));
        }
        let key = self.prompt_key(lineage);
        if let Some(id) = self.entry_on(&key) {
            let upgraded = class == StateClass::Ordinary && self.make_ordinary(id);
            self.restamp(id);
            return Ok(Inserted {
                id,
                new: false,
                upgraded,
            });
        }
        let mark = key.chain.last().copied().unwrap_or_default();
        let checkpoint = class == StateClass::Checkpoint && !self.in_use(mark);
        let state_bytes = state
            .state_bytes()
            .saturating_add(lineage_bytes(row, key.chain.len()));
        let id = self.next_id;
        self.next_id += 1;
        self.publications += 1;
        let stamp = self.recency.next();
        self.entries.push(Entry {
            id,
            block_size: self.block_size,
            salt: lineage.salt,
            tokens: lineage.tokens.to_vec(),
            chain: key.chain,
            state,
            state_bytes,
            last_used: stamp,
            checkpoint,
            publication: self.publications,
        });
        Ok(Inserted {
            id,
            new: true,
            upgraded: false,
        })
    }

    fn entry_on(&self, key: &PromptKey<'_>) -> Option<EntryId> {
        self.entries
            .iter()
            .find(|old| {
                old.salt == key.salt
                    && old.block_size == key.block_size
                    && old.chain == key.chain
                    && old.tokens[..] == key.tokens[..]
            })
            .map(Entry::id)
    }

    /// The block `lineage` ends on and the state there, if any (Splash
    /// `Cache::blockAt`, `StateCache::stateResident`).
    pub(crate) fn locate(&self, lineage: Lineage<'_>) -> (u64, Option<EntryId>) {
        let key = self.prompt_key(lineage);
        (
            key.chain.last().copied().unwrap_or_default(),
            self.entry_on(&key),
        )
    }

    /// Reuses entry `id` and restamps it; an ordinary reuse upgrades a
    /// checkpoint, a checkpoint never downgrades an ordinary state (Splash
    /// `StateCache::reuseStoredState`). False when it is gone.
    pub(crate) fn reuse(&mut self, id: EntryId, checkpoint: bool) -> bool {
        if self.get(id).is_none() {
            return false;
        }
        if !checkpoint {
            self.make_ordinary(id);
        }
        self.restamp(id);
        true
    }

    /// True when `id` was a checkpoint (Splash `StateCache::makeOrdinary`:
    /// the publication stays).
    fn make_ordinary(&mut self, id: EntryId) -> bool {
        self.get_mut(id)
            .is_some_and(|entry| std::mem::replace(&mut entry.checkpoint, false))
    }

    fn restamp(&mut self, id: EntryId) {
        let stamp = self.recency.next();
        if let Some(entry) = self.get_mut(id) {
            entry.last_used = stamp;
        }
    }

    /// The deepest state `b < |P|` whose lineage equals `P[..b]`; on a tie
    /// the most recently used (Splash `StateCache::acquireDeepest`).
    pub(crate) fn lookup(&self, prompt: Lineage<'_>) -> Option<StoreHit> {
        let key = self.prompt_key(prompt);
        self.entries
            .iter()
            .filter(|entry| entry.resumes(&key))
            .max_by_key(|entry| (entry.depth(), entry.last_used))
            .map(|entry| StoreHit {
                id: entry.id,
                depth: entry.depth(),
            })
    }

    /// [`Self::lookup`] among states at most `depth` deep: the GDN half of a
    /// composite state serves a request that needs no draft (Splash
    /// `QwenStateStorage::restore` with `restoreDraftState` false).
    pub(crate) fn lookup_within(&self, prompt: Lineage<'_>, depth: usize) -> Option<StoreHit> {
        let key = self.prompt_key(prompt);
        self.entries
            .iter()
            .filter(|entry| entry.depth() <= depth && entry.resumes(&key))
            .max_by_key(|entry| (entry.depth(), entry.last_used))
            .map(|entry| StoreHit {
                id: entry.id,
                depth: entry.depth(),
            })
    }

    /// Stamps `id` newest (Splash `StateCache::release` after a restore).
    pub(crate) fn touch(&mut self, id: EntryId) {
        self.restamp(id);
    }

    /// An unfinished turn resumes from the state at `block`, published or
    /// yet to be; a checkpoint there becomes ordinary (Splash
    /// `StateCache::useState`). Recency is set when the last use ends.
    pub(crate) fn use_state(&mut self, block: u64) {
        let uses = self.uses.entry(block).or_insert(0);
        *uses += 1;
        if *uses == 1 {
            for entry in self
                .entries
                .iter_mut()
                .filter(|entry| entry.mark() == block)
            {
                entry.checkpoint = false;
            }
        }
    }

    /// Ends one use of `block`; at the last, its state becomes the newest of
    /// its class (Splash `StateCache::unuse`).
    pub(crate) fn end_use(&mut self, block: u64) {
        let Some(uses) = self.uses.get_mut(&block) else {
            return;
        };
        *uses -= 1;
        if *uses > 0 {
            return;
        }
        self.uses.remove(&block);
        if let Some(id) = self
            .entries
            .iter()
            .find(|entry| entry.mark() == block)
            .map(Entry::id)
        {
            self.restamp(id);
        }
    }

    pub(crate) fn in_use(&self, block: u64) -> bool {
        self.uses.contains_key(&block)
    }

    /// `id` and its publication while it is a checkpoint (Splash
    /// `StateCache::checkpointState`).
    pub(crate) fn checkpoint_state(&self, id: EntryId) -> Option<(EntryId, u64)> {
        self.get(id)
            .filter(|entry| entry.checkpoint)
            .map(|entry| (entry.id, entry.publication))
    }

    /// Splash `StateCache::checkpointPinned`: no restore holds a lease on a
    /// stored state, so a checkpoint is never pinned.
    pub(crate) fn checkpoint_pinned(&self, _id: EntryId, _publication: u64) -> bool {
        false
    }

    /// Some entry sits on a strict descendant of `block` (Splash
    /// `KvCache::stateBelow`).
    pub(crate) fn state_below(&self, block: u64) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.chain[..entry.chain.len().saturating_sub(1)].contains(&block))
    }

    /// The block hashes of `lineage`'s whole blocks, as the store keys them.
    pub(crate) fn chain(&self, lineage: Lineage<'_>) -> Vec<u64> {
        block_chain(lineage, self.block_size)
    }

    /// Drops one entry; frees no block (its chain becomes dead leaves once
    /// the caller re-syncs the marks, Splash `StateCache::erase`). A use of
    /// its block stays.
    pub(crate) fn remove(&mut self, id: EntryId) -> bool {
        let before = self.entries.len();
        self.entries.retain(|entry| entry.id != id);
        self.entries.len() < before
    }

    /// The block of every entry.
    pub(crate) fn marks(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.iter().map(Entry::mark)
    }

    /// Drops every entry and use (`ResetScope::Command`, a new pool).
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.uses.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    type Log = Rc<RefCell<Vec<String>>>;

    struct TestState {
        name: String,
        bytes: u64,
        log: Log,
    }

    impl StateBytes for TestState {
        fn state_bytes(&self) -> u64 {
            self.bytes
        }
    }

    impl Drop for TestState {
        fn drop(&mut self) {
            self.log.borrow_mut().push(format!("drop {}", self.name));
        }
    }

    type Store = HybridStateStore<TestState>;

    const BS: u32 = 16;
    const ANCHOR: u64 = 60;

    fn lineage(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 100_000 + i).collect()
    }

    fn fork(base: &[u32], at: usize, len: usize, seed: u32) -> Vec<u32> {
        let mut tokens = base[..at].to_vec();
        tokens.extend((0..(len - at) as u32).map(|i| seed * 100_000 + 50_000 + i));
        tokens
    }

    struct Fixture {
        log: Log,
    }

    impl Fixture {
        fn new() -> Self {
            Self { log: Rc::default() }
        }

        fn state(&self, name: &str) -> TestState {
            TestState {
                name: name.to_owned(),
                bytes: ANCHOR,
                log: self.log.clone(),
            }
        }

        fn drain(&self) -> Vec<String> {
            std::mem::take(&mut *self.log.borrow_mut())
        }
    }

    fn hit(store: &Store, prompt: &[u32], salt: u64) -> Option<usize> {
        store
            .lookup(Lineage::text(prompt, salt))
            .map(|hit| hit.depth)
    }

    #[test]
    fn deepest_exact_resume_across_states() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 100);
        let b = lineage(2, 300);
        store
            .insert(
                Lineage::text(&a[..64], 0),
                fx.state("a64"),
                StateClass::Ordinary,
            )
            .unwrap();
        store
            .insert(
                Lineage::text(&b[..160], 0),
                fx.state("b160"),
                StateClass::Ordinary,
            )
            .unwrap();
        store
            .insert(
                Lineage::text(&b[..240], 0),
                fx.state("b240"),
                StateClass::Ordinary,
            )
            .unwrap();

        assert_eq!(hit(&store, &fork(&a, 80, 120, 3), 0), Some(64));
        assert_eq!(hit(&store, &fork(&a, 40, 120, 3), 0), None);
        assert_eq!(hit(&store, &fork(&b, 250, 400, 4), 0), Some(240));
        assert_eq!(hit(&store, &fork(&b, 200, 400, 4), 0), Some(160));
        // `b < |P|`: a prompt equal to the lineage cannot resume at its end.
        assert_eq!(hit(&store, &b[..240], 0), Some(160));
    }

    #[test]
    fn equal_depth_prefers_recent() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let base = lineage(1, 64);
        let id = store
            .insert(
                Lineage::text(&base, 0),
                fx.state("c64"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        let prompt = [base.clone(), vec![1, 2]].concat();
        let found = store.lookup(Lineage::text(&prompt, 0)).unwrap();
        assert_eq!((found.id, found.depth), (id, 64));

        let other = store
            .insert(
                Lineage::text(&base, 9),
                fx.state("c64 salt 9"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        assert_ne!(other, id);
        assert_eq!(store.lookup(Lineage::text(&prompt, 9)).unwrap().id, other);
    }

    #[test]
    fn salt_isolates_lineages() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 100);
        store
            .insert(
                Lineage::text(&a[..96], 7),
                fx.state("a"),
                StateClass::Ordinary,
            )
            .unwrap();
        assert_eq!(hit(&store, &a, 7), Some(96));
        assert_eq!(hit(&store, &a, 0), None);
        assert_eq!(hit(&store, &a, 8), None);
    }

    #[test]
    fn extra_keys_split_lineages_with_equal_tokens() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 100);
        let image = vec![vec![], vec![42]];
        let keyed = Lineage {
            tokens: &a[..32],
            extra_keys: &image,
            salt: 0,
        };
        store
            .insert(keyed, fx.state("a32"), StateClass::Ordinary)
            .unwrap();
        store
            .insert(
                Lineage::text(&a[..16], 0),
                fx.state("a16"),
                StateClass::Ordinary,
            )
            .unwrap();
        let probe = Lineage {
            tokens: &a,
            extra_keys: &image,
            salt: 0,
        };
        assert_eq!(store.lookup(probe).map(|hit| hit.depth), Some(32));
        // Without the image key only the block before it still lines up.
        assert_eq!(hit(&store, &a, 0), Some(16));
    }

    #[test]
    fn block_size_32_keys_match_the_allocator_chain() {
        let fx = Fixture::new();
        let mut store = Store::new(32);
        let a = lineage(1, 100);
        let id = store
            .insert(
                Lineage::text(&a[..64], 5),
                fx.state("a"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        let expected =
            compute_paged_prefix_block_hashes(&a, 64, 32, &vec![Vec::new(); 2], 5).unwrap();
        let entry = store.get(id).unwrap();
        assert_eq!(entry.chain(), &expected[..]);
        assert_eq!(entry.mark(), expected[1]);
        assert!(
            store
                .insert(
                    Lineage::text(&a[..48], 5),
                    fx.state("bad"),
                    StateClass::Ordinary
                )
                .is_err(),
            "48 is a 16-boundary, not a 32-boundary"
        );
        assert_eq!(hit(&store, &fork(&a, 70, 120, 2), 5), Some(64));
        assert_eq!(hit(&store, &fork(&a, 60, 120, 2), 5), None);
    }

    #[test]
    fn hash_collision_with_different_tokens_misses() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 100);
        let b = lineage(2, 100);
        store
            .insert(
                Lineage::text(&a[..64], 0),
                fx.state("a"),
                StateClass::Ordinary,
            )
            .unwrap();
        // Forge a collision: A's entry now carries B's block hashes.
        store.entries[0].chain = block_chain(Lineage::text(&b[..64], 0), BS);
        assert_eq!(hit(&store, &b, 0), None);
    }

    /// A state sits only on a complete KV block (Splash `Cache::blockAt`).
    /// Mutation PB1: `insert` skips the alignment check → red.
    #[test]
    fn a_state_sits_only_on_a_whole_block() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 100);
        for row in [0, 40, 99] {
            let error = store
                .insert(
                    Lineage::text(&a[..row], 0),
                    fx.state("x"),
                    StateClass::Ordinary,
                )
                .err()
                .map(|error| error.reason);
            assert!(
                error
                    .as_deref()
                    .is_some_and(|reason| reason.contains("is not a complete KV block")),
                "row {row}: {error:?}"
            );
        }
        assert_eq!(store.bytes().entries, 0);
        store
            .insert(
                Lineage::text(&a[..96], 0),
                fx.state("a"),
                StateClass::Ordinary,
            )
            .unwrap();
        assert_eq!(store.bytes().entries, 1);
    }

    /// One state per block: a second publish on the same block of the same
    /// lineage reuses the first and restamps it (Splash
    /// `reuseCompositeState`); another salt is another block.
    /// Mutation DD1: no dedup in `insert` → red (two entries).
    #[test]
    fn a_second_state_on_one_block_reuses_the_first() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 192);
        let first = store
            .insert(
                Lineage::text(&a, 0),
                fx.state("first"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        let stamp = store.get(first).unwrap().last_used();
        let again = store
            .insert(
                Lineage::text(&a, 0),
                fx.state("again"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        assert_eq!(again, first);
        assert_eq!(store.bytes().entries, 1);
        assert!(store.get(first).unwrap().last_used() > stamp);
        assert_eq!(fx.drain(), vec!["drop again".to_owned()]);
        store
            .insert(
                Lineage::text(&a, 3),
                fx.state("salted"),
                StateClass::Ordinary,
            )
            .unwrap();
        assert_eq!(store.bytes().entries, 2);
    }

    /// The last use of a block restamps its state.
    #[test]
    fn end_use_restamps_and_units_go_oldest_first() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = store
            .insert(
                Lineage::text(&lineage(1, 32), 0),
                fx.state("a"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        let b = store
            .insert(
                Lineage::text(&lineage(2, 32), 0),
                fx.state("b"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        let a_mark = store.get(a).unwrap().mark();
        let oldest = |store: &Store| {
            store
                .entries()
                .min_by_key(|entry| entry.last_used())
                .map(|entry| entry.id())
        };
        assert_eq!(oldest(&store), Some(a));
        store.use_state(a_mark);
        store.use_state(a_mark);
        store.end_use(a_mark);
        assert!(store.in_use(a_mark));
        assert_eq!(oldest(&store), Some(a));
        store.end_use(a_mark);
        assert!(!store.in_use(a_mark));
        assert_eq!(oldest(&store), Some(b));
        assert!(store.remove(b));
        assert!(!store.remove(b));
        assert_eq!(store.marks().count(), 1);
    }

    #[test]
    fn inserting_leaves_cache_limit_totals_unchanged() {
        let fx = Fixture::new();
        let before = crate::cache_limit::coordinator().registered_pool_bytes();
        let mut store = Store::new(BS);
        let a = lineage(1, 64);
        store
            .insert(Lineage::text(&a, 0), fx.state("a"), StateClass::Ordinary)
            .unwrap();
        assert_eq!(
            crate::cache_limit::coordinator().registered_pool_bytes(),
            before
        );
    }

    fn class_of(store: &Store, id: EntryId) -> StateClass {
        if store.get(id).unwrap().checkpoint() {
            StateClass::Checkpoint
        } else {
            StateClass::Ordinary
        }
    }

    /// Splash `StateCache`: an ordinary publication or a use upgrades a
    /// checkpoint and keeps its publication; a checkpoint never downgrades
    /// an ordinary state; a checkpoint at a block in use is ordinary.
    /// Mutation UP1 (a reuse keeps the old class) → red.
    #[test]
    fn ordinary_publication_upgrades_a_checkpoint() {
        use StateClass::{Checkpoint, Ordinary};
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 64);
        let first = store
            .insert(Lineage::text(&a, 0), fx.state("cp"), Checkpoint)
            .unwrap();
        assert!(first.new);
        assert_eq!(class_of(&store, first.id), Checkpoint);
        let publication = store.get(first.id).unwrap().publication();
        assert_eq!(
            store.checkpoint_state(first.id),
            Some((first.id, publication))
        );
        assert!(!store.checkpoint_pinned(first.id, publication));
        let again = store
            .insert(Lineage::text(&a, 0), fx.state("ord"), Ordinary)
            .unwrap();
        assert_eq!(
            again,
            Inserted {
                id: first.id,
                new: false,
                upgraded: true
            }
        );
        assert_eq!(store.bytes().entries, 1);
        assert_eq!(class_of(&store, first.id), Ordinary);
        assert_eq!(store.get(first.id).unwrap().publication(), publication);
        assert_eq!(store.checkpoint_state(first.id), None);

        let down = store
            .insert(Lineage::text(&a, 0), fx.state("cp again"), Checkpoint)
            .unwrap();
        assert!(!down.new && !down.upgraded);
        assert_eq!(class_of(&store, first.id), Ordinary);

        let b = lineage(2, 64);
        let used = store
            .insert(Lineage::text(&b, 0), fx.state("b"), Checkpoint)
            .unwrap()
            .id;
        assert_ne!(store.get(used).unwrap().publication(), publication);
        store.use_state(store.get(used).unwrap().mark());
        assert_eq!(class_of(&store, used), Ordinary);

        let c = lineage(3, 64);
        let c_mark = block_chain(Lineage::text(&c, 0), BS)
            .last()
            .copied()
            .unwrap();
        store.use_state(c_mark);
        let in_use = store
            .insert(Lineage::text(&c, 0), fx.state("c"), Checkpoint)
            .unwrap()
            .id;
        assert_eq!(class_of(&store, in_use), Ordinary);
        assert!(store.reuse(in_use, true));
        assert!(!store.reuse(999, false));
    }

    /// An entry's bytes count its lineage's token and chain vectors
    /// (03-PLAN "count the token vectors").
    /// Mutation TB1 (lineage bytes not added) → red.
    #[test]
    fn state_bytes_count_the_lineage() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let short: Vec<u32> = (0..1_600).collect();
        let long: Vec<u32> = (0..78_592).collect();
        let a = store
            .insert(
                Lineage::text(&short, 0),
                fx.state("short"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        let b = store
            .insert(
                Lineage::text(&long, 0),
                fx.state("long"),
                StateClass::Ordinary,
            )
            .unwrap()
            .id;
        assert_eq!(store.get(a).unwrap().state_bytes(), ANCHOR + 6_400 + 800);
        assert_eq!(
            store.get(b).unwrap().state_bytes(),
            ANCHOR + 314_368 + 39_296
        );
        assert_eq!(
            store.bytes().state_bytes,
            2 * ANCHOR + 6_400 + 800 + 314_368 + 39_296
        );
    }

    /// Splash `KvCache::stateBelow` looks at strict ancestors only.
    /// Mutation SB1 (the entry's own block counts) → red.
    #[test]
    fn state_below_sees_strict_ancestors_only() {
        let fx = Fixture::new();
        let mut store = Store::new(BS);
        let a = lineage(1, 48);
        let chain = block_chain(Lineage::text(&a, 0), BS);
        assert_eq!(chain.len(), 3);
        store
            .insert(Lineage::text(&a, 0), fx.state("a"), StateClass::Ordinary)
            .unwrap();
        assert!(store.state_below(chain[0]));
        assert!(store.state_below(chain[1]));
        assert!(!store.state_below(chain[2]));
        assert!(!store.state_below(0x5eed));
    }

    #[test]
    fn new_store_keeps_the_block_size() {
        assert_eq!(Store::new(16).block_size(), 16);
        assert_eq!(Store::new(0).block_size(), 1);
    }
}
