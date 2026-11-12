//! Content-keyed resume states for DFlash2 turns (Splash `StateCache`).
//!
//! An entry is one state on one complete KV block: the GDN cell and draft
//! window at a block-aligned row of a token lineage. It holds no target KV:
//! the KV is the prefix cache's chain, which the entry keeps resident by
//! marking the block it sits on (Splash `StateMarks`). Lookup keys on the
//! paged allocator's chained block hashes plus an exact token compare and the
//! cache salt; the owner id plays no part. State bytes go to one state
//! budget.
//!
//! Every byte the store holds is an MLX array the allocator already counts, so
//! nothing here registers with `cache_limit::coordinator()`.

use std::sync::Arc;

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::*;

use crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes;

pub(crate) const STATE_BUDGET_ENV: &str = "MLX_DFLASH2_STATE_BUDGET_MB";
pub(crate) const DEFAULT_STATE_BUDGET_MB: u64 = 3072;

const MIB: u64 = 1024 * 1024;

fn parse_state_budget(raw: Option<&str>) -> u64 {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_STATE_BUDGET_MB)
        .saturating_mul(MIB)
}

/// Bytes a parked state keeps alive, charged to the state budget.
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

pub(crate) struct Entry<S> {
    id: EntryId,
    block_size: u32,
    salt: u64,
    tokens: Vec<u32>,
    chain: Vec<u64>,
    state: S,
    state_bytes: u64,
    last_used: u64,
    uses: u32,
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

    pub(crate) fn state_bytes(&self) -> u64 {
        self.state_bytes
    }

    #[cfg(test)]
    pub(crate) fn last_used(&self) -> u64 {
        self.last_used
    }

    pub(crate) fn in_use(&self) -> bool {
        self.uses > 0
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StoreHit {
    pub(crate) id: EntryId,
    pub(crate) depth: usize,
}

impl StoreHit {
    /// The live state wins a tie.
    pub(crate) fn beats_live(&self, live_depth: Option<usize>) -> bool {
        live_depth.is_none_or(|live| self.depth > live)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StoreBytes {
    pub(crate) entries: usize,
    pub(crate) state_bytes: u64,
}

pub(crate) struct HybridStateStore<S> {
    block_size: u32,
    state_budget: u64,
    entries: Vec<Entry<S>>,
    next_id: EntryId,
    recency: Arc<CacheRecency>,
}

impl<S: StateBytes> HybridStateStore<S> {
    /// State budget from `MLX_DFLASH2_STATE_BUDGET_MB` (default 3072).
    pub(crate) fn new(block_size: u32) -> Self {
        let raw = std::env::var(STATE_BUDGET_ENV).ok();
        Self::with_state_budget(block_size, parse_state_budget(raw.as_deref()))
    }

    pub(crate) fn with_state_budget(block_size: u32, state_budget: u64) -> Self {
        Self {
            block_size: block_size.max(1),
            state_budget,
            entries: Vec::new(),
            next_id: 1,
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

    #[cfg(test)]
    pub(crate) fn state_budget(&self) -> u64 {
        self.state_budget
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
    /// same block of the same lineage is reused and restamped instead
    /// (Splash `reuseCompositeState`: one state per block). Then the oldest
    /// entries no turn uses go until the state budget holds the rest.
    /// Returns the id and whether it is new.
    pub(crate) fn insert(&mut self, lineage: Lineage<'_>, state: S) -> Result<(EntryId, bool)> {
        let row = lineage.tokens.len();
        let bs = self.block_size as usize;
        if row == 0 || !row.is_multiple_of(bs) {
            return Err(Error::from_reason(format!(
                "DFlash2 state at row {row} is not a complete KV block (block size {bs})"
            )));
        }
        let state_bytes = state.state_bytes();
        if state_bytes > self.state_budget {
            return Err(Error::from_reason(format!(
                "DFlash2 state of {state_bytes} B exceeds the {} B state budget",
                self.state_budget
            )));
        }
        let key = self.prompt_key(lineage);
        let stamp = self.recency.next();
        if let Some(old) = self.entries.iter_mut().find(|old| {
            old.salt == key.salt
                && old.block_size == key.block_size
                && old.chain == key.chain
                && old.tokens[..] == key.tokens[..]
        }) {
            old.last_used = stamp;
            return Ok((old.id, false));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry {
            id,
            block_size: self.block_size,
            salt: lineage.salt,
            tokens: lineage.tokens.to_vec(),
            chain: key.chain,
            state,
            state_bytes,
            last_used: stamp,
            uses: 0,
        });
        self.trim_to_budget();
        Ok((id, true))
    }

    /// Drops the oldest entries no turn uses until the state bytes fit the
    /// budget; returns how many went.
    pub(crate) fn trim_to_budget(&mut self) -> usize {
        let mut dropped = 0;
        while self.bytes().state_bytes > self.state_budget {
            let Some((_, id)) = self.oldest_unit(|entry| !entry.in_use()) else {
                break;
            };
            self.remove(id);
            dropped += 1;
        }
        dropped
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

    /// Stamps `id` newest (Splash `StateCache::release` after a restore).
    pub(crate) fn touch(&mut self, id: EntryId) {
        let stamp = self.recency.next();
        if let Some(entry) = self.get_mut(id) {
            entry.last_used = stamp;
        }
    }

    /// A running turn resumes from `id` (Splash `StateCache::useState`).
    pub(crate) fn use_entry(&mut self, id: EntryId) {
        if let Some(entry) = self.get_mut(id) {
            entry.uses += 1;
        }
    }

    /// Ends one use of `id`; the state becomes the newest of its class
    /// (Splash `StateCache::unuse`).
    pub(crate) fn unuse(&mut self, id: EntryId) {
        let stamp = self.recency.next();
        if let Some(entry) = self.get_mut(id) {
            entry.uses = entry.uses.saturating_sub(1);
            entry.last_used = stamp;
        }
    }

    /// The oldest entry `ok` accepts, with its stamp (one clock; a whole
    /// state is one unit).
    pub(crate) fn oldest_unit(&self, ok: impl Fn(&Entry<S>) -> bool) -> Option<(u64, EntryId)> {
        self.entries
            .iter()
            .filter(|entry| ok(entry))
            .map(|entry| (entry.last_used, entry.id))
            .min()
    }

    /// Drops one entry; frees no block (its chain becomes dead leaves once
    /// the caller re-syncs the marks, Splash `StateCache::erase`).
    pub(crate) fn remove(&mut self, id: EntryId) -> bool {
        let before = self.entries.len();
        self.entries.retain(|entry| entry.id != id);
        self.entries.len() < before
    }

    /// The block of every entry.
    pub(crate) fn marks(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.iter().map(Entry::mark)
    }

    /// Drops every entry (`ResetScope::Command`, a new pool).
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
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
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let b = lineage(2, 300);
        store
            .insert(Lineage::text(&a[..64], 0), fx.state("a64"))
            .unwrap();
        store
            .insert(Lineage::text(&b[..160], 0), fx.state("b160"))
            .unwrap();
        store
            .insert(Lineage::text(&b[..240], 0), fx.state("b240"))
            .unwrap();

        assert_eq!(hit(&store, &fork(&a, 80, 120, 3), 0), Some(64));
        assert_eq!(hit(&store, &fork(&a, 40, 120, 3), 0), None);
        assert_eq!(hit(&store, &fork(&b, 250, 400, 4), 0), Some(240));
        assert_eq!(hit(&store, &fork(&b, 200, 400, 4), 0), Some(160));
        // `b < |P|`: a prompt equal to the lineage cannot resume at its end.
        assert_eq!(hit(&store, &b[..240], 0), Some(160));
    }

    #[test]
    fn equal_depth_prefers_recent_and_live_wins_ties() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let base = lineage(1, 64);
        let id = store
            .insert(Lineage::text(&base, 0), fx.state("c64"))
            .unwrap()
            .0;
        let prompt = [base.clone(), vec![1, 2]].concat();
        let found = store.lookup(Lineage::text(&prompt, 0)).unwrap();
        assert_eq!((found.id, found.depth), (id, 64));
        assert!(!found.beats_live(Some(64)));
        assert!(found.beats_live(Some(63)));
        assert!(found.beats_live(None));

        let other = store
            .insert(Lineage::text(&base, 9), fx.state("c64 salt 9"))
            .unwrap()
            .0;
        assert_ne!(other, id);
        assert_eq!(store.lookup(Lineage::text(&prompt, 9)).unwrap().id, other);
    }

    #[test]
    fn salt_isolates_lineages() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        store
            .insert(Lineage::text(&a[..96], 7), fx.state("a"))
            .unwrap();
        assert_eq!(hit(&store, &a, 7), Some(96));
        assert_eq!(hit(&store, &a, 0), None);
        assert_eq!(hit(&store, &a, 8), None);
    }

    #[test]
    fn extra_keys_split_lineages_with_equal_tokens() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let image = vec![vec![], vec![42]];
        let keyed = Lineage {
            tokens: &a[..32],
            extra_keys: &image,
            salt: 0,
        };
        store.insert(keyed, fx.state("a32")).unwrap();
        store
            .insert(Lineage::text(&a[..16], 0), fx.state("a16"))
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
        let mut store = Store::with_state_budget(32, u64::MAX);
        let a = lineage(1, 100);
        let id = store
            .insert(Lineage::text(&a[..64], 5), fx.state("a"))
            .unwrap()
            .0;
        let expected =
            compute_paged_prefix_block_hashes(&a, 64, 32, &vec![Vec::new(); 2], 5).unwrap();
        let entry = store.get(id).unwrap();
        assert_eq!(entry.chain(), &expected[..]);
        assert_eq!(entry.mark(), expected[1]);
        assert!(
            store
                .insert(Lineage::text(&a[..48], 5), fx.state("bad"))
                .is_err(),
            "48 is a 16-boundary, not a 32-boundary"
        );
        assert_eq!(hit(&store, &fork(&a, 70, 120, 2), 5), Some(64));
        assert_eq!(hit(&store, &fork(&a, 60, 120, 2), 5), None);
    }

    #[test]
    fn hash_collision_with_different_tokens_misses() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let b = lineage(2, 100);
        store
            .insert(Lineage::text(&a[..64], 0), fx.state("a"))
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
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        for row in [0, 40, 99] {
            let error = store
                .insert(Lineage::text(&a[..row], 0), fx.state("x"))
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
            .insert(Lineage::text(&a[..96], 0), fx.state("a"))
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
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 192);
        let first = store
            .insert(Lineage::text(&a, 0), fx.state("first"))
            .unwrap()
            .0;
        let stamp = store.get(first).unwrap().last_used();
        let again = store
            .insert(Lineage::text(&a, 0), fx.state("again"))
            .unwrap()
            .0;
        assert_eq!(again, first);
        assert_eq!(store.bytes().entries, 1);
        assert!(store.get(first).unwrap().last_used() > stamp);
        assert_eq!(fx.drain(), vec!["drop again".to_owned()]);
        store
            .insert(Lineage::text(&a, 3), fx.state("salted"))
            .unwrap();
        assert_eq!(store.bytes().entries, 2);
    }

    /// Over the state budget the oldest entry no turn uses goes first.
    /// Mutation BG1: `insert` skips `trim_to_budget` → red (three entries).
    #[test]
    fn state_budget_drops_the_oldest_state() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, ANCHOR * 5 / 2);
        let a = lineage(1, 64);
        let b = lineage(2, 64);
        let c = lineage(3, 64);
        let a_id = store.insert(Lineage::text(&a, 0), fx.state("a")).unwrap().0;
        let b_id = store.insert(Lineage::text(&b, 0), fx.state("b")).unwrap().0;
        store.use_entry(a_id);
        let c_id = store.insert(Lineage::text(&c, 0), fx.state("c")).unwrap().0;
        let ids: Vec<EntryId> = store.entries().map(Entry::id).collect();
        assert_eq!(ids, vec![a_id, c_id], "the in-use entry stays");
        assert_eq!(fx.drain(), vec!["drop b".to_owned()]);
        assert!(store.get(b_id).is_none());
        assert_eq!(store.bytes().state_bytes, 2 * ANCHOR);
        let too_big = TestState {
            name: "big".into(),
            bytes: ANCHOR * 3,
            log: fx.log.clone(),
        };
        assert!(store.insert(Lineage::text(&b, 0), too_big).is_err());
    }

    /// `unuse` ends the use and makes the state the newest of its class;
    /// the oldest unit skips what `ok` refuses.
    #[test]
    fn unuse_restamps_and_units_go_oldest_first() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = store
            .insert(Lineage::text(&lineage(1, 32), 0), fx.state("a"))
            .unwrap()
            .0;
        let b = store
            .insert(Lineage::text(&lineage(2, 32), 0), fx.state("b"))
            .unwrap()
            .0;
        assert_eq!(store.oldest_unit(|_| true).map(|(_, id)| id), Some(a));
        store.use_entry(a);
        assert_eq!(
            store.oldest_unit(|entry| !entry.in_use()).map(|(_, id)| id),
            Some(b)
        );
        store.unuse(a);
        assert!(!store.get(a).unwrap().in_use());
        assert_eq!(store.oldest_unit(|_| true).map(|(_, id)| id), Some(b));
        assert!(store.remove(b));
        assert!(!store.remove(b));
        assert_eq!(store.marks().count(), 1);
    }

    #[test]
    fn inserting_leaves_cache_limit_totals_unchanged() {
        let fx = Fixture::new();
        let before = crate::cache_limit::coordinator().registered_pool_bytes();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 64);
        store.insert(Lineage::text(&a, 0), fx.state("a")).unwrap();
        assert_eq!(
            crate::cache_limit::coordinator().registered_pool_bytes(),
            before
        );
    }

    #[test]
    fn state_budget_reads_megabytes_with_a_3072_default() {
        assert_eq!(parse_state_budget(None), 3072 * MIB);
        assert_eq!(parse_state_budget(Some("512")), 512 * MIB);
        assert_eq!(parse_state_budget(Some(" 0 ")), 0);
        assert_eq!(parse_state_budget(Some("lots")), 3072 * MIB);
        let raw = std::env::var(STATE_BUDGET_ENV).ok();
        let store = Store::new(16);
        assert_eq!(store.state_budget(), parse_state_budget(raw.as_deref()));
        assert_eq!(store.block_size(), 16);
    }
}
