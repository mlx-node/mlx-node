//! A state on a part page on the state tier: the key of its object, what a
//! resume on it needs, and the chain a turn on it leaves behind.

use std::sync::{Arc, Mutex};

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PageLayout};

use super::tests::{
    TestTier, attach_tier, drained, insert_state, kv_key, persistent_test_tier, window,
    writable_anchor,
};
use super::*;
use crate::engine::build_paged_extra_keys;
use crate::models::qwen3_5::dflash2_state::store::StoreHit;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::transformer::part_page_tests::{Rows, write_turn_rows};

const PAGE: usize = 32;
const PAGES: usize = 2;
const DEPTH: usize = PAGES * PAGE + PART_PAGE_ROWS;
const PROMPT: usize = 120;
const BLOCKS: u32 = 12;
const LIMIT: u64 = 1 << 30;

fn prompt() -> Vec<u32> {
    (0..PROMPT as u32).map(|i| (i * 7 + 3) % 15 + 1).collect()
}

/// A tiny model whose pool has 32-token int8 pages with keys token-major,
/// its states in 32-token blocks on `tier`, and the rows of its pages.
fn model_on(tier: TestTier) -> Option<(Qwen35Inner, TestTier, Rows)> {
    use crate::models::qwen3_5::dflash2_decode::tests::{
        metal_is_default_device, tiny_dflash_inner_with_config,
    };
    if !metal_is_default_device() {
        return None;
    }
    let mut inner = tiny_dflash_inner_with_config(0x5713_4801, |_| {});
    inner.initialize_paged_adapter().expect("a pool");
    let mut config = inner
        .paged_adapter
        .take()
        .expect("a pool")
        .layer_kv_pool()
        .config()
        .clone();
    config.block_size = PAGE as u32;
    config.max_seq_len = Some(BLOCKS * PAGE as u32);
    let rows = Rows {
        layers: config.num_layers as usize,
        heads: config.num_kv_heads as usize,
        dim: config.head_size as usize,
    };
    let layout = PageLayout {
        keys_token_major: true,
        ..PageLayout::default()
    };
    let pool = LayerKVPool::new_with_layout(config, BLOCKS, BLOCKS, MetalDtype::Int8, layout)
        .expect("an int8 pool");
    let allocator = Arc::new(Mutex::new(BlockAllocator::with_recency(
        BLOCKS,
        BLOCKS,
        PAGE as u32,
        Arc::clone(&inner.cache_recency),
    )));
    let mut adapter =
        PagedKVCacheAdapter::new(allocator, Arc::new(pool), PAGE as u32).expect("an adapter");
    adapter.set_cold_tier(state_context(&tier.tier));
    inner.dflash2_states = DFlash2States::new(PAGE as u32, Arc::clone(&inner.cache_recency));
    let window = window(&inner);
    attach_tier(
        &mut inner.dflash2_states,
        &inner.config,
        &adapter,
        &tier,
        window.nbytes(),
        window.slots(),
    );
    inner.paged_adapter = Some(adapter);
    inner.sync_gdn_state_marks();
    Some((inner, tier, rows))
}

/// The keys of the fragment and of the state object of the state `DEPTH`
/// rows deep on `tokens`.
fn part_page_state_keys(tier: &TestTier, tokens: &[u32]) -> (ColdCacheKey, ColdCacheKey) {
    let lineage = Lineage::text(&tokens[..DEPTH], 0);
    let fragment = part_page_keys(
        ColdGroup::Kv,
        tier.tier.kv_fingerprint,
        lineage,
        PAGE as u32,
    )
    .1;
    let state = part_page_keys(
        ColdGroup::DFlash2State,
        tier.tier.state_fingerprint,
        lineage,
        PAGE as u32,
    )
    .1;
    (fragment.expect("a part page"), state.expect("a part page"))
}

/// Puts the state `DEPTH` rows deep on `tokens` on the tier as the move of a
/// 16-token store leaves it: its two pages, the fragment of its last 16
/// rows, and its object under the key chained over those rows.
fn write_part_page_state(inner: &Qwen35Inner, tier: &TestTier, rows: Rows, tokens: &[u32]) {
    let manager = &tier.tier.manager;
    let kv = tier.tier.kv_fingerprint;
    let lineage = Lineage::text(&tokens[..DEPTH], 0);
    let (pages, _) = part_page_keys(ColdGroup::Kv, kv, lineage, PAGE as u32);
    let (fragment, state) = part_page_state_keys(tier, tokens);
    assert_eq!(pages.len(), PAGES);
    for (page, &key) in pages.iter().enumerate() {
        let block = rows.block(
            key,
            kv,
            &tokens[page * PAGE..(page + 1) * PAGE],
            page * PAGE,
        );
        assert!(manager.enqueue(block).expect("a page"));
        drained(tier);
    }
    let block = rows.block(fragment, kv, &tokens[PAGES * PAGE..DEPTH], PAGES * PAGE);
    assert!(manager.enqueue(block).expect("the fragment"));
    drained(tier);

    let states = &inner.dflash2_states;
    let disk = states.disk.as_ref().expect("a tier");
    let anchor = writable_anchor(&inner.config, window(inner));
    let (done, landed) = std::sync::mpsc::channel();
    let object = ColdStateObject {
        key: state,
        fingerprint: tier.tier.state_fingerprint,
        layout: disk.layout(DEPTH).expect("a layout"),
        label: ColdStateLabel {
            tokens: tokens[..DEPTH].to_vec(),
            extra_keys: Vec::new(),
            salt: 0,
            last_used: 9,
            checkpoint: false,
        },
        payload: encode_anchor(states, &inner.config, &anchor).expect("the state's bytes"),
    };
    assert!(manager.enqueue_state(object, MakeRoom::Evict, done).is_ok());
    drained(tier);
    drop(landed);
}

/// The deepest stored state a turn of `tokens` resumes on within `depth`.
fn resume_hit(inner: &Qwen35Inner, tokens: &[u32], depth: usize) -> Option<StoreHit> {
    inner
        .dflash2_states
        .store
        .deepest_accepted_within(Lineage::text(tokens, 0), depth, |id| restorable(inner, id))
}

/// A turn of `tokens` as sequence `seq` that resumes on the state `depth`
/// deep: the rows its prefix covers and the blocks it holds.
fn restore(inner: &mut Qwen35Inner, seq: u32, tokens: &[u32], depth: usize) -> (u32, Vec<u32>) {
    let keys = build_paged_extra_keys(tokens.len(), PAGE as u32, &[]);
    let adapter = inner.paged_adapter.as_mut().expect("a pool");
    adapter.arm_state_restore(depth as u32);
    let plan = adapter
        .prepare_turn_per_block_booking_hit_only(
            seq,
            tokens,
            tokens.len() as u32,
            &keys,
            0,
            false,
            depth as u32,
        )
        .expect("a prepared turn");
    let blocks = adapter
        .block_table()
        .expect("a block table")
        .blocks()
        .iter()
        .map(|block| block.block_id)
        .collect();
    (plan.cached_prefix_len, blocks)
}

/// The object of a state on a part page has the key chained over its last
/// 16 rows, never the key of its whole blocks (a whole state of those
/// blocks owns that one). A turn resumes on it only while its fragment is
/// on the tier, with a RAM copy too: those rows are in no block.
#[test]
fn a_state_on_a_part_page_needs_its_fragment_and_keeps_its_own_key() {
    let Some((mut inner, tier, rows)) =
        model_on(persistent_test_tier("state-tier-part-page-key", LIMIT))
    else {
        return;
    };
    let tokens = prompt();
    write_part_page_state(&inner, &tier, rows, &tokens);
    let (fragment, state) = part_page_state_keys(&tier, &tokens);
    assert_eq!(adopt_all(&mut inner), 1);
    let hit = resume_hit(&inner, &tokens, PROMPT - 1).expect("the state resumes");
    assert_eq!(hit.depth, DEPTH);

    let states = &inner.dflash2_states;
    let entry = states.store.get(hit.id).expect("the entry");
    let disk = states.disk.as_ref().expect("a tier");
    let whole = chain_keys(ColdGroup::DFlash2State, tier.tier.state_fingerprint, entry)
        .last()
        .copied();
    assert!(whole.is_some() && whole != Some(state));
    assert!(entry.disk().is_some_and(|copy| copy.key == state));
    assert!(
        disk.state_key(entry) == Some(state),
        "a write of it takes the key its object has"
    );
    assert!(state_object_key(states, hit.id) == Some(state));

    let anchor = writable_anchor(&inner.config, window(&inner));
    assert!(promote(&mut inner, hit.id, anchor));
    assert!(restorable(&inner, hit.id));
    assert!(tier.tier.manager.remove(&fragment));
    assert!(
        !restorable(&inner, hit.id),
        "its last 16 rows are in no block and not on the tier"
    );
    assert_eq!(resume_hit(&inner, &tokens, PROMPT - 1), None);
}

/// A turn that resumes on a state on a part page fills the state's tail
/// block, and that block is then a block like any other: the state the turn
/// leaves has a chain of whole blocks up to its row, every one of them on
/// the tier once it is durable, the tail block with the fragment's rows and
/// the turn's. The next process resumes on that state with no fragment.
/// The state on the part page still resumes there, in a block of its own.
#[test]
fn a_turn_on_a_part_page_leaves_a_whole_chain_the_next_process_restores() {
    const TAG: &str = "state-tier-part-page-turn";
    let Some((mut inner, tier, rows)) = model_on(persistent_test_tier(TAG, LIMIT)) else {
        return;
    };
    let tokens = prompt();
    let end = (PAGES + 1) * PAGE;
    write_part_page_state(&inner, &tier, rows, &tokens);
    assert_eq!(adopt_all(&mut inner), 1);
    let hit = resume_hit(&inner, &tokens, PROMPT - 1).expect("the state resumes");
    assert_eq!(hit.depth, DEPTH);

    let (restored, blocks) = restore(&mut inner, 0, &tokens, DEPTH);
    assert_eq!(restored as usize, DEPTH);
    assert_eq!(blocks.len(), PAGES + 1);
    let keys = build_paged_extra_keys(tokens.len(), PAGE as u32, &[]);
    let adapter = inner.paged_adapter.as_mut().expect("a pool");
    let pool = Arc::clone(adapter.layer_kv_pool());
    adapter
        .confirm_aux_prefix_primed(DEPTH as u32)
        .expect("the state is installed");
    adapter
        .record_tokens(&tokens[DEPTH..end])
        .expect("the turn's rows");
    write_turn_rows(&pool, rows, blocks[PAGES], PAGES * PAGE, PART_PAGE_ROWS);
    adapter
        .register_committed_blocks(end, &keys, 0)
        .expect("the full blocks");

    let next = insert_state(&mut inner, &tokens[..end], StateClass::Ordinary);
    let entry = inner.dflash2_states.store.get(next).expect("the entry");
    assert_eq!(entry.chain().len() * PAGE, entry.depth());
    let adapter = inner.paged_adapter.as_ref().expect("a pool");
    assert!(
        entry
            .chain()
            .iter()
            .all(|&hash| adapter.prefix_cached(hash)),
        "every block of the new state's chain is a block of the pool's prefixes"
    );
    let mut status = persist(&mut inner, next);
    for _ in 0..8 {
        if status != PersistStatus::Started && status != PersistStatus::Busy {
            break;
        }
        drained(&tier);
        poll_offload(&mut inner);
        status = persist(&mut inner, next);
    }
    assert_eq!(
        status,
        PersistStatus::Durable,
        "the new state and its whole chain are on the tier"
    );
    let tail = tier
        .tier
        .manager
        .read_back(
            kv_key(&inner, &tier, next, PAGES),
            tier.tier.kv_fingerprint,
            &rows.layout(PAGE),
        )
        .expect("the tail block's object");
    let written: Vec<(Vec<u8>, Vec<u8>)> = tail
        .layers
        .into_iter()
        .map(|layer| (layer.keys, layer.values))
        .collect();
    assert!(
        written == rows.page(PAGES * PAGE, PAGE, PAGE),
        "the fragment's rows and the turn's"
    );

    drop(pool);
    drop(inner);
    let Some((mut inner, tier, rows)) = model_on(tier.reopen(TAG, LIMIT)) else {
        return;
    };
    assert_eq!(adopt_all(&mut inner), 2);
    let hit = resume_hit(&inner, &tokens, PROMPT - 1).expect("the new state resumes");
    assert_eq!(hit.depth, end);
    let reads = tier.tier.manager.stats().hits;
    let (restored, whole) = restore(&mut inner, 0, &tokens, end);
    assert_eq!(restored as usize, end);
    assert_eq!(whole.len(), PAGES + 1);
    assert_eq!(
        tier.tier.manager.stats().hits - reads,
        (PAGES + 1) as u64,
        "three blocks and no fragment"
    );
    let pool = Arc::clone(
        inner
            .paged_adapter
            .as_ref()
            .expect("a pool")
            .layer_kv_pool(),
    );
    assert!(
        pool.read_block_all_layers(whole[PAGES]).expect("the block")
            == rows.page(PAGES * PAGE, PAGE, PAGE)
    );

    let hit = resume_hit(&inner, &tokens, end - 1).expect("the state on the part page resumes");
    assert_eq!(hit.depth, DEPTH);
    let (restored, part) = restore(&mut inner, 1, &tokens, DEPTH);
    assert_eq!(restored as usize, DEPTH);
    assert_eq!(part[..PAGES], whole[..PAGES], "the pages are shared");
    assert_ne!(part[PAGES], whole[PAGES], "not the block of the whole page");
    assert!(
        pool.read_block_all_layers(part[PAGES]).expect("the block")
            == rows.page(PAGES * PAGE, PART_PAGE_ROWS, PAGE)
    );
}
