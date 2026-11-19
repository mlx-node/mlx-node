//! A state on a part page through the adapter: its whole pages come back by
//! the chain restore, its last 16 rows from the state's fragment into one
//! block of the request alone.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{
    BlockAllocator, ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout,
    ColdCacheManager, ColdGroup, ColdLayerBlock, LayerKVPool, PageLayout, PagedAttentionConfig,
    PrefixKeys, RestoreUploadBudget,
};

use super::cold_tier::{ChainPrefetchPoll, ChainPrefetchStart};
use super::paged_kv_cache_adapter::{ColdContextBacking, ColdTierContext, PagedKVCacheAdapter};

const LAYERS: usize = 2;
const HEADS: usize = 2;
const DIM: usize = 32;
const PAGE: usize = 32;
const HALF: usize = PAGE / 2;
const PAGES: usize = 2;
const DEPTH: usize = PAGES * PAGE + HALF;
const PROMPT: usize = 120;
const BLOCKS: u32 = 12;
const QUOTA: u64 = 16 << 20;
const SECOND: u128 = 1_000_000_000;

fn keys_token_major() -> PageLayout {
    PageLayout {
        keys_token_major: true,
        ..PageLayout::default()
    }
}

/// Rows of an int8 page with keys token-major, each byte a function of its
/// row in the whole lineage: K `[KV][rows][D]`, V `[KV][D][rows]`, each
/// side's f32 scales `[KV][rows]` after its plane.
#[derive(Clone, Copy)]
pub(crate) struct Rows {
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) dim: usize,
}

const ROWS: Rows = Rows {
    layers: LAYERS,
    heads: HEADS,
    dim: DIM,
};

impl Rows {
    fn side(self, rows: usize) -> usize {
        self.heads * self.dim * rows + self.heads * rows * 4
    }

    /// The layout of a block of `rows` rows.
    pub(crate) fn layout(self, rows: usize) -> ColdCacheLayout {
        ColdCacheLayout {
            block_size: rows as u32,
            num_layers: self.layers as u32,
            num_kv_heads: self.heads as u32,
            head_size: self.dim as u32,
            cache_dtype: "Int8".to_string(),
            key_bytes_per_layer: self.side(rows),
            value_bytes_per_layer: self.side(rows),
        }
    }

    /// One layer of `rows` rows from lineage row `first` on a page of
    /// `page_rows` rows; the rows past `rows` zero. No byte of a written
    /// row is zero.
    pub(crate) fn layer(
        self,
        layer: usize,
        first: usize,
        rows: usize,
        page_rows: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let plane = self.heads * self.dim * page_rows;
        let byte = |value: usize, head: usize, row: usize, d: usize| {
            (1 + (row * 7 + d * 3 + head * 5 + layer * 11 + value * 13) % 250) as u8
        };
        let scale = |value: usize, head: usize, row: usize| {
            let seed = (row * 3 + head * 7 + layer * 13 + value * 17) % 200;
            [
                1 + seed as u8,
                2 + (seed % 100) as u8,
                3 + (seed % 50) as u8,
                0x3f,
            ]
        };
        let mut keys = vec![0u8; self.side(page_rows)];
        let mut values = vec![0u8; self.side(page_rows)];
        for head in 0..self.heads {
            for row in 0..rows {
                for d in 0..self.dim {
                    keys[(head * page_rows + row) * self.dim + d] = byte(0, head, first + row, d);
                    values[(head * self.dim + d) * page_rows + row] = byte(1, head, first + row, d);
                }
                let at = plane + (head * page_rows + row) * 4;
                keys[at..at + 4].copy_from_slice(&scale(0, head, first + row));
                values[at..at + 4].copy_from_slice(&scale(1, head, first + row));
            }
        }
        (keys, values)
    }

    /// Every layer of [`Self::layer`].
    pub(crate) fn page(
        self,
        first: usize,
        rows: usize,
        page_rows: usize,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        (0..self.layers)
            .map(|layer| self.layer(layer, first, rows, page_rows))
            .collect()
    }

    /// The block of `tokens` (lineage rows from `first`) under `key`.
    pub(crate) fn block(
        self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        tokens: &[u32],
        first: usize,
    ) -> ColdCacheBlock {
        let rows = tokens.len();
        ColdCacheBlock {
            key,
            fingerprint,
            tokens: tokens.to_vec(),
            layers: self
                .page(first, rows, rows)
                .into_iter()
                .map(|(keys, values)| ColdLayerBlock { keys, values })
                .collect(),
            layout: self.layout(rows),
        }
    }
}

/// Writes rows `from..page_rows` of the page that starts at lineage row
/// `first` into `block`, as a turn's forward does; its rows below `from`
/// stay.
pub(crate) fn write_turn_rows(
    pool: &LayerKVPool,
    rows: Rows,
    block: u32,
    first: usize,
    from: usize,
) {
    let page_rows = pool.block_size() as usize;
    let mut held = pool.read_block_all_layers(block).expect("the block");
    for (layer, (keys, values)) in held.iter_mut().enumerate() {
        let (full_keys, full_values) = rows.layer(layer, first, page_rows, page_rows);
        let (kept_keys, kept_values) = rows.layer(layer, first, from, page_rows);
        for at in 0..keys.len() {
            if kept_keys[at] == 0 {
                keys[at] = full_keys[at];
            }
            if kept_values[at] == 0 {
                values[at] = full_values[at];
            }
        }
    }
    let layers: Vec<(&[u8], &[u8])> = held
        .iter()
        .map(|(keys, values)| (keys.as_slice(), values.as_slice()))
        .collect();
    pool.write_block_all_layers(block, &layers)
        .expect("the turn's rows");
}

/// A tier with the two pages and the fragment of an 80-row state.
struct Store {
    root: PathBuf,
    fingerprint: ColdCacheFingerprint,
    tokens: Vec<u32>,
    /// The prompt's image keys per page.
    keys: Vec<Vec<u64>>,
    /// The image keys of the fragment's 16 rows.
    part_keys: Vec<u64>,
}

impl Drop for Store {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

impl Store {
    fn new(tag: &str) -> Self {
        Self::with_keys(tag, vec![Vec::new(); PROMPT.div_ceil(PAGE)], Vec::new())
    }

    fn with_keys(tag: &str, keys: Vec<Vec<u64>>, part_keys: Vec<u64>) -> Self {
        let root = std::env::temp_dir().join(format!(
            "mlx-part-page-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let store = Self {
            root,
            fingerprint: ColdCacheFingerprint::from_components([b"part-page".as_slice()]),
            tokens: (0..PROMPT as u32).map(|i| 500 + i * 13).collect(),
            keys,
            part_keys,
        };
        let manager = store.manager(QUOTA);
        for page in 0..PAGES {
            store.write(
                &manager,
                ROWS.block(
                    store.page_key(page),
                    store.fingerprint,
                    &store.tokens[page * PAGE..(page + 1) * PAGE],
                    page * PAGE,
                ),
            );
        }
        store.write(&manager, store.fragment());
        store
    }

    fn manager(&self, quota: u64) -> Arc<ColdCacheManager> {
        Arc::new(ColdCacheManager::open_at(self.root.clone(), quota, 0, 8).expect("a cold cache"))
    }

    fn write(&self, manager: &ColdCacheManager, block: ColdCacheBlock) {
        assert!(
            manager.enqueue(block).expect("a block"),
            "the queue took it"
        );
        assert!(manager.drain(Duration::from_secs(10)));
    }

    fn page_key(&self, page: usize) -> ColdCacheKey {
        let mut parent = None;
        for index in 0..=page {
            parent = Some(ColdCacheKey::chain(
                ColdGroup::Kv,
                self.fingerprint,
                parent,
                &self.tokens[index * PAGE..(index + 1) * PAGE],
                &self.keys[index],
                0,
                index,
            ));
        }
        parent.expect("a page")
    }

    fn fragment_key(&self) -> ColdCacheKey {
        ColdCacheKey::chain(
            ColdGroup::Kv,
            self.fingerprint,
            Some(self.page_key(PAGES - 1)),
            &self.tokens[PAGES * PAGE..DEPTH],
            &self.part_keys,
            0,
            PAGES,
        )
    }

    fn fragment(&self) -> ColdCacheBlock {
        ROWS.block(
            self.fragment_key(),
            self.fingerprint,
            &self.tokens[PAGES * PAGE..DEPTH],
            PAGES * PAGE,
        )
    }

    /// A state-backed adapter over the tier on a fresh pool of `layout`.
    fn adapter(
        &self,
        manager: &Arc<ColdCacheManager>,
        layout: PageLayout,
    ) -> (PagedKVCacheAdapter, Arc<LayerKVPool>) {
        let config = PagedAttentionConfig {
            block_size: PAGE as u32,
            gpu_memory_mb: 256,
            head_size: DIM as u32,
            num_kv_heads: HEADS as u32,
            num_layers: LAYERS as u32,
            max_seq_len: Some(BLOCKS * PAGE as u32),
            max_batch_size: Some(1),
        };
        let pool = Arc::new(
            LayerKVPool::new_with_layout(config, BLOCKS, BLOCKS, MetalDtype::Int8, layout)
                .expect("an int8 pool"),
        );
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(BLOCKS, BLOCKS, PAGE as u32)));
        let mut adapter =
            PagedKVCacheAdapter::new(allocator, Arc::clone(&pool), PAGE as u32).expect("adapter");
        adapter.set_cold_tier(ColdTierContext {
            manager: Arc::clone(manager),
            fingerprint: self.fingerprint,
            backing: ColdContextBacking::StateOnly,
        });
        (adapter, pool)
    }

    /// A turn of the whole prompt that resumes on the state `depth` deep;
    /// the rows its prefix covers.
    fn resume(&self, adapter: &mut PagedKVCacheAdapter, seq: u32, depth: usize) -> u32 {
        self.resume_within(adapter, seq, depth, depth)
    }

    /// [`Self::resume`] whose lookup reads `lookup` rows of the prompt.
    fn resume_within(
        &self,
        adapter: &mut PagedKVCacheAdapter,
        seq: u32,
        depth: usize,
        lookup: usize,
    ) -> u32 {
        self.resume_with(adapter, seq, depth, lookup, &self.keys)
    }

    /// [`Self::resume_within`] of a prompt with the image keys `keys`.
    fn resume_with(
        &self,
        adapter: &mut PagedKVCacheAdapter,
        seq: u32,
        depth: usize,
        lookup: usize,
        keys: &[Vec<u64>],
    ) -> u32 {
        adapter.arm_state_restore(depth as u32);
        adapter
            .prepare_turn_per_block_booking_hit_only(
                seq,
                &self.tokens,
                PROMPT as u32,
                keys,
                0,
                false,
                lookup as u32,
            )
            .expect("a prepared turn")
            .cached_prefix_len
    }

    fn page_hash(&self, page: usize) -> u64 {
        PrefixKeys::PerBlock(&self.keys).chain_hashes(&self.tokens[..(page + 1) * PAGE], 32, 0)
            [page]
    }
}

fn block_ids(adapter: &PagedKVCacheAdapter) -> Vec<u32> {
    adapter
        .block_table()
        .expect("a block table")
        .blocks()
        .iter()
        .map(|block| block.block_id)
        .collect()
}

fn tail_refs(adapter: &PagedKVCacheAdapter) -> u32 {
    adapter.block_table().expect("a block table").blocks()[PAGES].get_ref_count()
}

/// The fragment's rows in the first half of a page, the other half zero.
fn widened() -> Vec<(Vec<u8>, Vec<u8>)> {
    ROWS.page(PAGES * PAGE, HALF, PAGE)
}

/// A state of two pages and 16 rows restores at 80 rows: the pages by the
/// chain restore, the 16 rows from its fragment into one block more, the
/// block's other half zero. While that block holds less than a page it is
/// the request's alone and no prefix names it: a second request that
/// restores the same state shares the pages and takes its own block with
/// its own read of the fragment. Once the first request fills its block, the
/// block is registered like any full block.
#[test]
fn a_part_page_comes_back_into_a_block_of_the_request_alone() {
    let store = Store::new("alone");
    let manager = store.manager(QUOTA);
    let (mut adapter, pool) = store.adapter(&manager, keys_token_major());

    // The lookup reads the tail block's whole page of the prompt.
    assert_eq!(
        store.resume_within(&mut adapter, 1, DEPTH, PROMPT - 1),
        DEPTH as u32
    );
    assert!(adapter.state_restored());
    assert!(
        adapter.aux_prefix_unbacked(),
        "the state's GDN half is owed"
    );
    assert_eq!(adapter.current_token_count(), DEPTH as u32);
    assert_eq!(adapter.request_tokens(), &store.tokens[..DEPTH]);
    let first = block_ids(&adapter);
    assert_eq!(first.len(), PAGES + 1, "the pages and one block more");
    assert_eq!(
        tail_refs(&adapter),
        1,
        "the tail block is the request's alone"
    );
    assert!(
        !adapter.prefix_cached(store.page_hash(PAGES)),
        "no prefix names a block of 16 rows"
    );
    assert_eq!(
        pool.read_block_all_layers(first[PAGES]).expect("the block"),
        widened(),
        "the fragment's rows, the other half zero"
    );
    for (page, &block) in first[..PAGES].iter().enumerate() {
        assert_eq!(
            pool.read_block_all_layers(block).expect("a page"),
            ROWS.page(page * PAGE, PAGE, PAGE)
        );
    }

    let hits = manager.stats().hits;
    assert_eq!(store.resume(&mut adapter, 2, DEPTH), DEPTH as u32);
    let second = block_ids(&adapter);
    assert_eq!(second[..PAGES], first[..PAGES], "the pages are shared");
    assert_ne!(
        second[PAGES], first[PAGES],
        "each request has its own block"
    );
    assert_eq!(tail_refs(&adapter), 1);
    assert_eq!(
        manager.stats().hits - hits,
        1,
        "the second request read the fragment again, and no page"
    );

    adapter.activate_request(1).expect("the first request");
    adapter
        .confirm_aux_prefix_primed(DEPTH as u32)
        .expect("the state is installed");
    adapter
        .register_committed_blocks(DEPTH, &store.keys, 0)
        .expect("the full blocks");
    assert!(
        !adapter.prefix_cached(store.page_hash(PAGES)),
        "a block of 16 rows is not a full block"
    );
    adapter
        .record_tokens(&store.tokens[DEPTH..(PAGES + 1) * PAGE])
        .expect("the turn's rows");
    write_turn_rows(&pool, ROWS, first[PAGES], PAGES * PAGE, HALF);
    assert_eq!(
        pool.read_block_all_layers(first[PAGES]).expect("the block"),
        ROWS.page(PAGES * PAGE, PAGE, PAGE)
    );
    assert_eq!(
        pool.read_block_all_layers(second[PAGES])
            .expect("the block"),
        widened(),
        "the first request's rows are in its own block"
    );
    adapter
        .register_committed_blocks((PAGES + 1) * PAGE, &store.keys, 0)
        .expect("the full blocks");
    assert!(
        adapter.prefix_cached(store.page_hash(PAGES)),
        "a full block is a block like any other"
    );
}

/// The prompt's rows past a part-page restore at 80 rows take only the
/// blocks the 120 rows still need: the restored table holds 3 blocks of 32
/// (96 rows), so one block more, not the 2 that 40 rows would take from a
/// block edge.
/// Mutation: the count starts at the cached rows, not the table's blocks →
/// red (2 blocks).
#[test]
fn the_rows_past_a_part_page_book_only_the_blocks_they_need() {
    let store = Store::new("suffix");
    let manager = store.manager(QUOTA);
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(
        store.resume_within(&mut adapter, 1, DEPTH, PROMPT - 1),
        DEPTH as u32
    );
    assert_eq!(block_ids(&adapter).len(), PAGES + 1);
    let booked = adapter
        .allocate_suffix_blocks(PROMPT as u32)
        .expect("the prompt's blocks");
    assert_eq!(booked, 1, "one block more");
    assert_eq!(block_ids(&adapter).len(), PROMPT.div_ceil(PAGE));
}

/// The fragment is taken only by a pool whose page has keys token-major,
/// only for a state half a page past its chain, and only as the object its
/// own chain names with the store's identity and the fragment's layout. A
/// vLLM block of 16 tokens has that layout too: it carries its own store's
/// identity, and its pool is never one with keys token-major.
#[test]
fn a_fragment_is_taken_only_as_the_part_page_of_its_own_chain() {
    let store = Store::new("own");
    let manager = store.manager(QUOTA);
    let fragment = store.fragment_key();
    let before = manager.modified(&fragment).expect("the fragment");
    let rows = &store.tokens[PAGES * PAGE..DEPTH];

    let (mut vllm, _pool) = store.adapter(&manager, PageLayout::default());
    let restored = store.resume(&mut vllm, 1, DEPTH) as usize;
    assert!(
        restored <= PAGES * PAGE,
        "a vLLM page takes no fragment ({restored} rows)"
    );
    assert!(vllm.block_table().expect("a table").num_blocks() <= PAGES);
    drop(vllm);
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(
        store.resume(&mut adapter, 1, PAGES * PAGE + 8),
        (PAGES * PAGE) as u32,
        "only a state half a page past its chain has a fragment"
    );
    assert_eq!(manager.modified(&fragment), Some(before), "nothing read it");
    drop(adapter);

    // The same rows and layout under the fragment's name, of another store.
    let other = ColdCacheFingerprint::from_components([b"a 16-token store".as_slice()]);
    assert!(manager.remove(&fragment));
    store.write(&manager, ROWS.block(fragment, other, rows, PAGES * PAGE));
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(
        store.resume(&mut adapter, 1, DEPTH),
        (PAGES * PAGE) as u32,
        "another store's block is no fragment"
    );
    assert_eq!(adapter.block_table().expect("a table").num_blocks(), PAGES);
    drop(adapter);

    // A whole page under the fragment's name.
    let _ = manager.remove(&fragment);
    store.write(
        &manager,
        ROWS.block(
            fragment,
            store.fingerprint,
            &store.tokens[PAGES * PAGE..(PAGES + 1) * PAGE],
            PAGES * PAGE,
        ),
    );
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(
        store.resume(&mut adapter, 1, DEPTH),
        (PAGES * PAGE) as u32,
        "a whole page is no fragment"
    );
    assert_eq!(adapter.block_table().expect("a table").num_blocks(), PAGES);
    drop(adapter);

    // Other rows than the prompt's under the fragment's name.
    let _ = manager.remove(&fragment);
    let other_rows: Vec<u32> = rows.iter().map(|token| token + 1).collect();
    store.write(
        &manager,
        ROWS.block(fragment, store.fingerprint, &other_rows, PAGES * PAGE),
    );
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(
        store.resume(&mut adapter, 1, DEPTH),
        (PAGES * PAGE) as u32,
        "a fragment of other rows"
    );
    assert_eq!(adapter.block_table().expect("a table").num_blocks(), PAGES);
}

/// The fragment of a chat with images is named by the image keys of its own
/// 16 rows. The prompt's keys of the page's later rows are no part of that
/// name; another image on the fragment's rows is another name.
#[test]
fn a_fragment_is_named_by_the_image_keys_of_its_own_rows() {
    let keys = vec![
        Vec::new(),
        vec![1, 2, 77, 3],
        vec![1, 2, 88, 5, 99, 20],
        Vec::new(),
    ];
    let store = Store::with_keys("images", keys, vec![1, 2, 88, 5]);
    let manager = store.manager(QUOTA);
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(store.resume(&mut adapter, 1, DEPTH), DEPTH as u32);
    drop(adapter);

    for page in [vec![1, 2, 88, 6, 99, 20], vec![1, 2, 99, 20], Vec::new()] {
        let mut other = store.keys.clone();
        other[PAGES] = page;
        let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
        assert_eq!(
            store.resume_with(&mut adapter, 1, DEPTH, DEPTH, &other),
            (PAGES * PAGE) as u32,
            "other image keys on the fragment's rows: {:?}",
            other[PAGES]
        );
    }
}

/// A restore uses the fragment as it uses the pages: its time moves to the
/// restore's. Under a lower limit the tier then gives up the fragment no
/// sooner than a page of its chain.
#[test]
fn a_restore_keeps_the_fragment_as_long_as_its_pages() {
    let store = Store::new("touch");
    let manager = store.manager(QUOTA);
    let fragment = store.fragment_key();
    let pages: Vec<ColdCacheKey> = (0..PAGES).map(|page| store.page_key(page)).collect();
    let now = manager.modified(&fragment).expect("the fragment");
    let old = now - 3_600 * SECOND;
    assert!(manager.set_modified(&fragment, old));
    for (index, page) in pages.iter().enumerate() {
        assert!(manager.set_modified(page, old + (index as u128 + 1) * SECOND));
    }

    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(store.resume(&mut adapter, 1, DEPTH), DEPTH as u32);
    let used = manager.modified(&fragment).expect("the fragment");
    let page_times: Vec<u128> = pages
        .iter()
        .map(|page| manager.modified(page).expect("a page"))
        .collect();
    drop(adapter);
    let total = manager.total_bytes();
    drop(manager);

    let cut = store.manager(total - 1);
    cut.fit_quota();
    let fragment_left = cut.contains_in(&fragment, ColdGroup::Kv);
    let pages_left = pages
        .iter()
        .filter(|page| cut.contains_in(page, ColdGroup::Kv))
        .count();
    assert!(
        usize::from(fragment_left) + pages_left <= PAGES,
        "the lower limit takes an object"
    );
    assert!(
        fragment_left || pages_left == 0,
        "the fragment went before its pages ({pages_left} of {PAGES} pages left)"
    );
    assert!(
        used > old + 1_800 * SECOND,
        "the restore used the fragment: its time moved"
    );
    assert!(
        page_times.iter().all(|&page| used >= page),
        "the fragment is used after its pages"
    );
}

/// The dense gather reads a vLLM page only: it names the page it refuses.
#[test]
fn the_dense_gather_refuses_a_keys_token_major_page() {
    let store = Store::new("gather");
    let manager = store.manager(QUOTA);
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    assert_eq!(store.resume(&mut adapter, 1, DEPTH), DEPTH as u32);
    let error = adapter
        .gather_kv_for_prefill_sdpa(0, PAGE as u32)
        .err()
        .expect("no dense gather over this page");
    assert!(
        error.contains("keys_token_major: true")
            && error.contains("is read by its model's own page gathers"),
        "the refusal names the page: {error}"
    );
    let error = adapter
        .gather_kv_for_dense_cache_hit_prefill(0, PAGE as u32)
        .err()
        .expect("no dense gather over this page");
    assert!(
        error.contains("keys_token_major: true"),
        "the refusal names the page: {error}"
    );
}

/// A chain read beside other turns keeps the state's fragment on the tier
/// until the landing's lookup reads it; that lookup stands at the state's
/// row.
#[test]
fn a_chain_prefetch_holds_the_fragment_for_its_landing() {
    let store = Store::new("prefetch");
    let manager = store.manager(QUOTA);
    let fragment = store.fragment_key();
    let (mut adapter, _pool) = store.adapter(&manager, keys_token_major());
    let start = adapter.begin_chain_prefetch(
        &store.tokens,
        (PROMPT - 1) as u32,
        DEPTH,
        PrefixKeys::PerBlock(&store.keys),
        0,
        None,
    );
    let ChainPrefetchStart::Started(mut prefetch) = start else {
        panic!("the chain is on the tier");
    };
    assert!(manager.is_held_for_test(&fragment));
    let landed = loop {
        match adapter.poll_chain_prefetch(&mut prefetch, &mut RestoreUploadBudget::per_pass()) {
            ChainPrefetchPoll::Pending => std::thread::sleep(Duration::from_millis(2)),
            other => break other,
        }
    };
    assert!(
        matches!(landed, ChainPrefetchPoll::Landed { blocks: PAGES, .. }),
        "{landed:?}"
    );
    adapter.arm_landed_chain(&prefetch);
    assert_eq!(store.resume(&mut adapter, 1, DEPTH), DEPTH as u32);
    assert_eq!(tail_refs(&adapter), 1);
    drop(prefetch);
    assert!(!manager.is_held_for_test(&fragment));
}
