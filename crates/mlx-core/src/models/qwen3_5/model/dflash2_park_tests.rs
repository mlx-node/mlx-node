//! The take order a refused paged step reclaims through: ladder and history
//! entries, marked chains and image entries.

use std::sync::Arc;

use super::*;
use crate::engine::backend::{ChatBackend, WholeTurnArgs};
use crate::engine::governor::ledger::UnitKey;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::dflash2_decode::tests::{
    metal_is_default_device, tiny_dflash_inner_with_config,
};
use crate::models::qwen3_5::dflash2_turn::{
    GrowReclaim, Keep, claimed, clear_grow_reclaim_log, grow_reclaim_log, paged_step_claimed,
};

fn fixture(seed: u64) -> Qwen35Inner {
    tiny_dflash_inner_with_config(seed, |_| {})
}

fn tokenizer() -> Arc<Qwen3Tokenizer> {
    crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer()
}

fn first_prompt(seed: u32) -> Vec<u32> {
    (0..40).map(|i| (seed + i * 7) % 15 + 1).collect()
}

fn ar_config() -> ChatConfig {
    ChatConfig {
        temperature: Some(0.0),
        max_new_tokens: Some(2),
        reasoning_effort: Some("none".to_string()),
        ..ChatConfig::default()
    }
}

fn run_paged_ar_turn(
    inner: &mut Qwen35Inner,
    tokenizer: &Arc<Qwen3Tokenizer>,
    tokens: &[u32],
) -> Result<()> {
    let config = ar_config();
    let params = ChatBackend::resolve_params(inner, &config);
    let thinking = ChatBackend::thinking_setup(inner, &config);
    let mut args = WholeTurnArgs {
        tokens,
        tokenizer,
        eos_id: u32::MAX,
        config: &config,
        params: &params,
        thinking,
        plan: TurnPlan {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            use_paged_attention: true,
            decoder: DecoderPlan::Autoregressive,
        },
        sink: None,
        cancelled: None,
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };
    ChatBackend::run_paged_turn(inner, &mut args)?;
    Ok(())
}

/// Forwards to the request's stream and calls `ChatStreamHandle::stop` on its
/// 20th delta, as the server's stop match does.
struct StopAt20<'a> {
    plumbing: &'a crate::engine::napi_glue::ChatStreamPlumbing,
    deltas: std::sync::Mutex<usize>,
}

impl crate::engine::backend::ChunkSink for StopAt20<'_> {
    fn send(&self, chunk: Result<crate::engine::types::ChatStreamChunk>) {
        if let Ok(delta) = &chunk
            && !delta.done
        {
            let mut deltas = self.deltas.lock().unwrap();
            *deltas += 1;
            if *deltas == 20 {
                self.plumbing.handle.stop();
            }
        }
        crate::engine::backend::ChunkSink::send(&self.plumbing.stream_tx, chunk);
    }
}

/// A turn stopped at its 20th token ends through the commit epilogue (Splash
/// `release` → `vacateLane`): the full blocks of the prompt and the reply up
/// to the stop are registered, so a new request's prefix hits through them.
///
/// Mutation: `finish_paged_turn` aborts a "cancelled" turn's pages in
/// place of registering them; the hit stops inside the prompt.
#[test]
fn a_stopped_turn_commits_through_the_match() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_stopped_turn_commits_through_the_match") {
        return Ok(());
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0x5730_0001);
    install_pool(&mut inner, 16, 16, passing_probe())?;
    let tokenizer = tokenizer();
    let prompt = first_prompt(3);
    let config = ChatConfig {
        max_new_tokens: Some(40),
        max_consecutive_tokens: Some(0),
        max_ngram_repeats: Some(0),
        ..ar_config()
    };
    let params = ChatBackend::resolve_params(&inner, &config);
    let thinking = ChatBackend::thinking_setup(&inner, &config);
    let (plumbing, mut rx) = crate::engine::napi_glue::chat_stream_plumbing();
    let sink = StopAt20 {
        plumbing: &plumbing,
        deltas: std::sync::Mutex::new(0),
    };
    ChatBackend::set_cache_owner_id(&mut inner, "stopped", None);
    let mut args = WholeTurnArgs {
        tokens: &prompt,
        tokenizer: &tokenizer,
        eos_id: u32::MAX,
        config: &config,
        params: &params,
        thinking,
        plan: TurnPlan {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            use_paged_attention: true,
            decoder: DecoderPlan::Autoregressive,
        },
        sink: Some(&sink),
        cancelled: Some(&plumbing.cancelled),
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };
    ChatBackend::run_paged_turn(&mut inner, &mut args)?;
    let mut done = None;
    while let Ok(chunk) = rx.try_recv() {
        let chunk = chunk?;
        if chunk.done {
            done = Some(chunk);
        }
    }
    let done = done.expect("the stopped turn sent no done chunk");
    assert_eq!(done.finish_reason.as_deref(), Some("stop"));
    let generated = done.num_tokens.unwrap() as usize;
    assert!((20..=21).contains(&generated), "{generated} tokens");
    let history = inner.cached_token_history.clone();
    assert_eq!(history.len(), prompt.len() + generated - 1);
    inner.discard_dense_paged_session();
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    let block_size = adapter.block_size() as usize;
    assert_eq!(block_size, 16);
    let mut next = history.clone();
    next.extend([1, 2, 3, 4, 5]);
    let (_, hit) = adapter.allocator().lock().unwrap().find_longest_cache_hit(
        &next,
        block_size as u32,
        &[],
        params.cache_salt,
    );
    assert_eq!(hit, history.len() / block_size * block_size);
    assert!(hit > prompt.len() / block_size * block_size);
    Ok(())
}

/// A pool starts at the runway and grows by Splash extents. Mutation: drop the `set_extent_growth` call in dense
/// `initialize_paged_adapter` → red (`None`).
#[test]
fn runway_pool_grows_by_splash_extents() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1606);
    inner.config.paged_cache_memory_mb = Some(256);
    inner.paged_adapter = None;
    inner.initialize_paged_adapter()?;
    let adapter = inner.paged_adapter.as_ref().expect("paged adapter");
    let blocks = adapter.block_telemetry().map_err(Error::from_reason)?;
    assert!(
        blocks.total_blocks < blocks.max_total_blocks,
        "the pool starts at the runway: {blocks:?}"
    );
    let bytes_per_block = adapter.bytes_per_block().map_err(Error::from_reason)?;
    assert_eq!(
        adapter.extent_growth_blocks(),
        Some(crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(bytes_per_block))
    );
    Ok(())
}

pub(crate) type Probe = Arc<dyn Fn(u32) -> std::result::Result<u32, String> + Send + Sync>;

/// Swaps the model's pool for one of `initial` blocks that grows to `max`,
/// its allocator on the model's clock, with `probe` as the grow probe.
pub(crate) fn install_pool(
    inner: &mut Qwen35Inner,
    initial: u32,
    max: u32,
    probe: Probe,
) -> Result<()> {
    inner.paged_adapter = None;
    inner.initialize_paged_adapter()?;
    let adapter = inner.paged_adapter.take().expect("paged adapter");
    let block_size = adapter.block_size();
    let pool = adapter.layer_kv_pool();
    let (config, dtype) = (pool.config().clone(), pool.cache_dtype());
    drop(adapter);
    let pool = mlx_paged_attn::LayerKVPool::new(config, initial, max, dtype)
        .map_err(Error::from_reason)?;
    let allocator = Arc::new(std::sync::Mutex::new(
        mlx_paged_attn::BlockAllocator::with_recency(
            initial,
            max,
            block_size,
            Arc::clone(&inner.cache_recency),
        ),
    ));
    let mut adapter = PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
        .map_err(Error::from_reason)?;
    adapter.set_grow_headroom_probe_override(Some(probe));
    inner.paged_adapter = Some(adapter);
    inner.sync_gdn_state_marks();
    Ok(())
}

pub(crate) fn passing_probe() -> Probe {
    Arc::new(Ok)
}

/// Declines (no room) until the reclaim log on this thread has `taken`
/// entries.
fn probe_after(taken: usize) -> Probe {
    Arc::new(move |blocks| {
        Ok(if grow_reclaim_log().len() >= taken {
            blocks
        } else {
            0
        })
    })
}

fn pool_block_size(inner: &Qwen35Inner) -> u32 {
    inner.paged_adapter.as_ref().expect("adapter").block_size()
}

pub(crate) fn chain_tokens(seed: u32, blocks: u32, block_size: u32) -> Vec<u32> {
    (0..blocks * block_size)
        .map(|i| (seed * 11 + i * 7) % 15 + 1)
        .collect()
}

/// Caches `tokens` (whole blocks) as a chain nothing holds; returns its
/// block hashes.
pub(crate) fn cache_chain(inner: &Qwen35Inner, tokens: &[u32]) -> Vec<u64> {
    cache_keyed_chain(inner, tokens, &[], 0)
}

/// [`cache_chain`] under per-block `keys` and `salt`.
fn cache_keyed_chain(
    inner: &Qwen35Inner,
    tokens: &[u32],
    keys: &[Vec<u64>],
    salt: u64,
) -> Vec<u64> {
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    let block_size = adapter.block_size();
    let prefix_keys = if keys.is_empty() {
        mlx_paged_attn::PrefixKeys::Uniform(&[])
    } else {
        mlx_paged_attn::PrefixKeys::PerBlock(keys)
    };
    let mut allocator = adapter.allocator().lock().expect("allocator");
    let blocks: Vec<_> = (0..tokens.len() as u32 / block_size)
        .map(|_| allocator.allocate().expect("a block for the chain"))
        .collect();
    allocator
        .cache_full_blocks_with_keys(tokens, &blocks, block_size, prefix_keys, salt)
        .expect("cache the chain");
    for block in blocks {
        allocator.free(block);
    }
    prefix_keys.chain_hashes(tokens, block_size, salt)
}

/// Sequence `seq_id` holds `blocks` uncached blocks.
fn hold_blocks(inner: &mut Qwen35Inner, seq_id: u32, blocks: u32) {
    let adapter = inner.paged_adapter.as_mut().expect("adapter");
    let block_size = adapter.block_size();
    adapter.begin_request(seq_id).expect("begin");
    adapter
        .allocate_suffix_blocks_for(seq_id, blocks * block_size)
        .expect("hold blocks");
}

/// A ladder rung over a cached `tokens` chain with no GDN arrays.
fn push_rung(inner: &mut Qwen35Inner, tokens: Vec<u32>, hashes: Vec<u64>, last_used: u64) {
    inner.push_ladder_entry_for_test(tokens.len() as u32, hashes, tokens, Vec::new(), last_used);
}

fn per_block_keys(tokens: &[u32], block_size: u32) -> Vec<Vec<u64>> {
    vec![Vec::new(); tokens.len() / block_size as usize]
}

/// A step that reserves blocks for `total` tokens of `seq_id`; counts its
/// runs.
fn reserve_step(
    runs: &mut u32,
    seq_id: u32,
    total: u32,
) -> impl FnMut(&mut Qwen35Inner) -> Result<()> + '_ {
    move |inner| {
        *runs += 1;
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .reserve_rows_for(seq_id, total)
            .map(|_| ())
            .map_err(Error::from_reason)
    }
}

/// At the cap a ladder entry's dropped marks make its chain evictable in the
/// same take, so the refused step runs on the retry.
/// Mutation: skip the mark sync in `drop_gdn_state` → red (the retry is
/// refused the same need and the met list stops the loop: `Err`).
#[test]
fn dropping_a_ladder_entry_makes_its_chain_evictable() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1801);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    hold_blocks(&mut inner, 1, 2);
    let chain = chain_tokens(1, 2, block_size);
    let hashes = cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, chain, hashes, stamp);
    clear_grow_reclaim_log();
    let mut runs = 0;
    paged_step_claimed(&mut inner, reserve_step(&mut runs, 1, 3 * block_size))?;
    assert_eq!(runs, 2);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    let mut allocator = adapter.allocator().lock().expect("allocator");
    let block = allocator
        .allocate()
        .expect("the other chain block is takeable");
    allocator.free(block);
    Ok(())
}

/// The model at the cap: sequence 9 holds a cached 2-block chain, the live
/// sequence 0 holds it too plus one uncached block, and a 1-block ladder
/// chain fills the pool. The prompt grow counts the live blocks as the
/// prompt's, so only the prepare's own refusal can reclaim.
fn cap_with_a_shared_live_table(seed: u64) -> Result<Qwen35Inner> {
    cap_with_a_shared_live_table_of(seed, 4)
}

/// [`cap_with_a_shared_live_table`] on a `total`-block pool: the shared
/// chain takes every block but the live one and the ladder one.
fn cap_with_a_shared_live_table_of(seed: u64, total: u32) -> Result<Qwen35Inner> {
    let mut inner = fixture(seed);
    install_pool(&mut inner, total, total, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let shared = chain_tokens(5, total - 2, block_size);
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.begin_request(9).map_err(Error::from_reason)?;
        adapter
            .allocate_suffix_blocks_for(9, shared.len() as u32)
            .map_err(Error::from_reason)?;
        adapter.record_tokens(&shared).map_err(Error::from_reason)?;
        adapter
            .register_full_blocks_for_reuse(&[], 0)
            .map_err(Error::from_reason)?;
        let mut live = shared.clone();
        live.push(1);
        let plan = adapter
            .prepare_turn(0, &live, live.len() as u32, true, &[], 0, false)
            .map_err(Error::from_reason)?;
        assert_eq!(plan.cached_prefix_len as usize, shared.len());
    }
    let rung = chain_tokens(6, 1, block_size);
    let hashes = cache_chain(&inner, &rung);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, rung, hashes, stamp);
    let blocks = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .block_telemetry()
        .map_err(Error::from_reason)?;
    assert_eq!((blocks.free_blocks, blocks.evictable_blocks), (0, 0));
    Ok(inner)
}

/// A prefix prepare at the cap whose only takeable chain is a ladder rung
/// the prompt does not share frees it and prepares.
/// Mutations: drop the `claimed` wrapper in dense `prime_prefix_state`
/// → red (`Err` "paged cache could not reserve"); drop the record after
/// the release in `prepare_turn_inner` → red (`Err`, no refusal).
#[test]
fn prefix_prepare_at_the_cap_frees_a_marked_chain() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = cap_with_a_shared_live_table(0xDFA5_1805)?;
    let block_size = pool_block_size(&inner);
    let prompt = chain_tokens(7, 2, block_size);
    clear_grow_reclaim_log();
    PagedBackend::prime_prefix_state(&mut inner, &prompt, true, block_size as usize, &[], 0)?;
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert!(inner.ladder_entries_for_test().is_empty());
    Ok(())
}

/// A prepare at the cap keeps the ladder rung its prompt restores through:
/// the newer, unrelated rung goes and the retried prepare hits the kept
/// chain.
/// Mutation: ignore `keep` in `gdn_unit_excluded` → red (the prompt's own
/// rung is dropped).
#[test]
fn a_prefix_prepare_keeps_its_own_ladder_entry() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1806);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let own = chain_tokens(8, 2, block_size);
    let own_hashes = cache_chain(&inner, &own);
    let other = chain_tokens(9, 2, block_size);
    let other_hashes = cache_chain(&inner, &other);
    let older = inner.cache_recency.next();
    push_rung(&mut inner, own.clone(), own_hashes.clone(), older);
    let newer = inner.cache_recency.next();
    push_rung(&mut inner, other, other_hashes, newer);
    let mut prompt = own.clone();
    prompt.extend(chain_tokens(10, 1, block_size));
    let keys = per_block_keys(&prompt, block_size);
    let keep = Keep {
        tokens: &prompt,
        keys: Some((&keys, 0)),
    };
    clear_grow_reclaim_log();
    let plan = claimed(&mut inner, keep, |inner| {
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                0,
                &prompt,
                prompt.len() as u32,
                true,
                &keys,
                0,
                false,
                prompt.len() as u32 - 1,
            )
            .map_err(Error::from_reason)
    })?;
    assert_eq!(plan.cached_prefix_len as usize, own.len());
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    let kept: Vec<u64> = inner
        .ladder_entries_for_test()
        .into_iter()
        .map(|(_, hash, _)| hash)
        .collect();
    assert_eq!(kept, vec![*own_hashes.last().expect("own final")]);
    Ok(())
}

/// The dense VLM prefix prepare at the cap frees a ladder chain its prompt
/// does not share.
/// Mutations: drop the `claimed` wrapper in
/// `prepare_dense_vlm_paged_prefix` → red (`Err`); drop the record after
/// the release in `prepare_turn_inner` → red (`Err`).
#[test]
fn vlm_prefix_prepare_at_the_cap_frees_a_ladder_chain() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = cap_with_a_shared_live_table(0xDFA5_1807)?;
    let block_size = pool_block_size(&inner);
    let prompt = chain_tokens(11, 2, block_size);
    let keys = per_block_keys(&prompt, block_size);
    clear_grow_reclaim_log();
    inner.prepare_dense_vlm_paged_prefix(
        &prompt,
        prompt.len() as u32,
        block_size,
        &keys,
        true,
        true,
        77,
        0,
    )?;
    assert_eq!(grow_reclaim_log().first(), Some(&GrowReclaim::Ladder));
    assert!(inner.ladder_entries_for_test().is_empty());
    Ok(())
}

/// Hashes are stale: a mark no entry holds keeps its chain out of the
/// takeable count until the sync before the count drops it, so the need is
/// met with no state taken.
/// Mutation: drop the mark sync at the top of `claimed` → red (the step
/// runs twice).
#[test]
fn stale_marks_are_resynced_before_the_count() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1808);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let stale = cache_chain(&inner, &chain_tokens(12, 2, block_size));
    let rung = chain_tokens(13, 2, block_size);
    let hashes = cache_chain(&inner, &rung);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, rung, hashes.clone(), stamp);
    let adapter = inner.paged_adapter.as_mut().expect("adapter");
    adapter.set_state_marks([stale[1], hashes[1]]);
    assert_eq!(adapter.takeable_blocks(), 0);
    adapter.begin_request(1).map_err(Error::from_reason)?;
    clear_grow_reclaim_log();
    let mut runs = 0;
    paged_step_claimed(&mut inner, reserve_step(&mut runs, 1, block_size))?;
    assert_eq!(runs, 1);
    assert!(grow_reclaim_log().is_empty());
    assert_eq!(inner.ladder_entries_for_test().len(), 1);
    Ok(())
}

/// At the cap only the deepest of one owner's nested rungs is a unit: the
/// shallower rung lies inside its chain and would free nothing.
/// Mutation: count a block for the first listed mark in
/// `BlockAllocator::freed_by_marks` → red (the older, shallower rung goes
/// first: `[Ladder, Ladder]`).
#[test]
fn cap_reclaim_takes_the_deepest_rung_only() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1810);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let chain = chain_tokens(16, 4, block_size);
    let hashes = cache_chain(&inner, &chain);
    let half = (2 * block_size) as usize;
    push_rung(&mut inner, chain[..half].to_vec(), hashes[..2].to_vec(), 10);
    push_rung(&mut inner, chain.clone(), hashes.clone(), 20);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .begin_request(1)
        .map_err(Error::from_reason)?;
    clear_grow_reclaim_log();
    let mut runs = 0;
    paged_step_claimed(&mut inner, reserve_step(&mut runs, 1, block_size))?;
    assert_eq!(runs, 2);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert_eq!(
        inner.ladder_entries_for_test(),
        vec![(2 * block_size, hashes[1], 10)]
    );
    let tips = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .allocator()
        .lock()
        .expect("allocator")
        .tip_marks();
    assert_eq!(tips, [hashes[1]].into_iter().collect());
    Ok(())
}

/// Splash `Cache::take`: a take that removed its unit is Met, whatever it
/// measured, so the ledger lists the model's units again. Parked state S
/// and the later rung L mark the same chain; L's arrays have another
/// holder. At the cap L holds the chain's pages and frees nothing
/// measurable (S still marks it); listed again, S holds them and its take
/// frees the block the step needs.
/// Mutation R2m (a take that measured nothing is Short) → red (`Err`: S
/// is never taken).
#[test]
fn a_take_that_frees_nothing_measurable_lists_again() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1811);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let chain = chain_tokens(17, 4, block_size);
    let hashes = cache_chain(&inner, &chain);
    park_state(&mut inner, &chain);
    let caches = owned_caches(&inner.config);
    let other_holder =
        super::state::clone_dense_linear_layer_caches(&inner.config, &caches).expect("ready");
    let stamp = inner.cache_recency.next();
    inner.push_ladder_entry_for_test(4 * block_size, hashes, chain, caches, stamp);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .begin_request(1)
        .map_err(Error::from_reason)?;
    clear_grow_reclaim_log();
    let mut runs = 0;
    paged_step_claimed(&mut inner, reserve_step(&mut runs, 1, block_size))?;
    assert_eq!(runs, 2);
    assert_eq!(
        grow_reclaim_log(),
        vec![GrowReclaim::Ladder, GrowReclaim::Store]
    );
    drop(other_holder);
    Ok(())
}

/// GDN caches a ladder rung can restore: the live state of a finished
/// DFlash2 turn.
fn restorable_caches(inner: &Qwen35Inner) -> Vec<Qwen3_5LayerCache> {
    super::state::clone_dense_linear_layer_caches(
        &inner.config,
        inner.caches.as_ref().expect("live caches"),
    )
    .expect("ready linear caches")
}

/// A fixture whose sequence 0 restored its prompt's GDN state from rung E
/// through the text or the VLM prepare; the hit stamped E, and rung F,
/// pushed after the prepare, is newer still. Returns E's final hash, F's,
/// and the prompt.
fn restored_from_rung(seed: u64, vlm: bool) -> Result<(Qwen35Inner, u64, u64, Vec<u32>)> {
    let tokenizer = tokenizer();
    let mut inner = fixture(seed);
    install_pool(&mut inner, 16, 16, passing_probe())?;
    run_paged_ar_turn(&mut inner, &tokenizer, &first_prompt(10))?;
    let caches = restorable_caches(&inner);
    // The pool turn leaves its own ladder and history entries; the rungs
    // below must be the only ones.
    inner.gdn_prefix_checkpoints.clear();
    inner.gdn_last_history_checkpoint = None;
    install_pool(&mut inner, 4, 8, probe_after(1))?;
    let block_size = pool_block_size(&inner);
    let rung = chain_tokens(17, 2, block_size);
    let hashes = mlx_paged_attn::chain_hashes(&rung, block_size, &[], 0);
    let e_final = hashes[1];
    let e_caches = restorable_caches_from(&inner, &caches);
    let f_caches = owned_caches(&inner.config);
    inner.push_ladder_entry_for_test(rung.len() as u32, hashes, rung.clone(), e_caches, 1);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .reset_for_new_request(0)
        .map_err(Error::from_reason)?;
    let mut prompt = rung.clone();
    prompt.extend(chain_tokens(19, 1, block_size));
    let keys = per_block_keys(&prompt, block_size);
    inner.cached_token_history.clear();
    if vlm {
        assert!(inner.prepare_dense_gdn_vlm_prefix_state(
            &prompt,
            rung.len() as u32,
            block_size,
            &keys,
            0,
            false,
            77,
        )?);
    } else {
        let prepared = inner.prepare_dense_gdn_prefix_state(
            &prompt,
            rung.len() as u32,
            block_size,
            &keys,
            0,
            false,
        )?;
        assert_eq!(prepared.restored_prefix_tokens as usize, rung.len());
    }
    let f_final = 0xF0F0;
    let f_stamp = inner.cache_recency.next();
    inner.push_ladder_entry_for_test(
        block_size,
        vec![f_final],
        chain_tokens(18, 1, block_size),
        f_caches,
        f_stamp,
    );
    Ok((inner, e_final, f_final, prompt))
}

/// GDN caches of their own: evaluated 4 MiB arrays no other state shares,
/// so a take measurably frees them.
pub(crate) fn owned_caches(config: &Qwen3_5Config) -> Vec<Qwen3_5LayerCache> {
    let mut caches = super::state::fresh_dense_layer_caches(config);
    for cache in &mut caches {
        if let Qwen3_5LayerCache::Linear(arrays) = cache {
            for slot in 0..arrays.len() {
                let array = MxArray::zeros(&[1, 1 << 20], None).expect("zeros");
                array.eval();
                arrays.set(slot, array).expect("set slot");
            }
        }
    }
    caches
}

fn restorable_caches_from(
    inner: &Qwen35Inner,
    caches: &[Qwen3_5LayerCache],
) -> Vec<Qwen3_5LayerCache> {
    super::state::clone_dense_linear_layer_caches(&inner.config, caches)
        .expect("ready linear caches")
}

fn rung_key(inner: &Qwen35Inner, prefix_len: u32, final_block_hash: u64) -> UnitKey {
    UnitKey::Ladder {
        owner_id: inner.active_cache_owner_id.clone(),
        prefix_len,
        final_block_hash,
    }
}

/// The rung a running turn restored from is InUse: a refused grow's bytes
/// pass over it, though it is older than rung F, for the text and the VLM
/// prepare. E shares the live caches and frees nothing; F's own arrays
/// cover the refusal.
/// Mutation: report no entry in use in dense `gdn_state_units` → red (E
/// goes first).
#[test]
fn a_replay_point_in_use_is_not_a_unit() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_replay_point_in_use_is_not_a_unit") {
        return Ok(());
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    for (seed, vlm) in [(0xDFA5_1811, false), (0xDFA5_1812, true)] {
        let (mut inner, e_final, f_final, _) = restored_from_rung(seed, vlm)?;
        inner.memory_registration =
            Some(crate::engine::governor::governor::tests::refusing_registration());
        let e_key = rung_key(&inner, 2 * pool_block_size(&inner), e_final);
        assert_eq!(inner.gdn_in_use_for_test(0), Some(&e_key), "vlm {vlm}");
        assert!(
            inner
                .gdn_state_units(Keep::default(), &Default::default())
                .iter()
                .any(|unit| !unit.in_use && unit.bytes > 0),
            "vlm {vlm}: the count must pass"
        );
        clear_grow_reclaim_log();
        let mut runs = 0;
        paged_step_claimed(&mut inner, |inner| {
            runs += 1;
            if runs == 1 {
                crate::engine::governor::governor::tests::refuse_one_byte(
                    inner.memory_registration.as_ref().expect("registered"),
                );
                inner
                    .paged_adapter
                    .as_ref()
                    .expect("adapter")
                    .refuse_for_test(0, 5);
                Err(Error::from_reason(
                    "context_length_exceeded: paged decode could not reserve 5 more block(s)",
                ))
            } else {
                Ok(())
            }
        })?;
        assert_eq!(runs, 2, "vlm {vlm}");
        assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder], "vlm {vlm}");
        let left: Vec<u64> = inner
            .ladder_entries_for_test()
            .into_iter()
            .map(|(_, hash, _)| hash)
            .collect();
        assert_eq!(left, vec![e_final], "vlm {vlm}: F {f_final:#x} went");
        assert_eq!(inner.gdn_in_use_for_test(0), Some(&e_key), "vlm {vlm}");
    }
    Ok(())
}

/// The serial turn's use end stamps its replay point after the turn end
/// stamped the table and after the history write.
/// Mutations: no stamp in `end_gdn_use` → red (E below the history); end
/// the use before the history write in
/// `remember_dense_gdn_history_checkpoint` → red (E below the history).
#[test]
fn use_end_stamps_the_replay_point_after_the_history() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "use_end_stamps_the_replay_point_after_the_history",
    ) {
        return Ok(());
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    let (mut inner, e_final, _, prompt) = restored_from_rung(0xDFA5_1813, false)?;
    let block_size = pool_block_size(&inner) as usize;
    let history = prompt[..2 * block_size].to_vec();
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter
            .allocate_suffix_blocks(history.len() as u32)
            .map_err(Error::from_reason)?;
        adapter
            .record_tokens(&history)
            .map_err(Error::from_reason)?;
        adapter
            .finalize_turn_keep_live(&[], 0)
            .map_err(Error::from_reason)?;
    }
    inner.cached_token_history = history;
    inner.remember_dense_gdn_history_checkpoint()?;
    let table = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .turn_end_stamp(0)
        .expect("turn end stamp");
    let written = inner.history_stamp_for_test().expect("history written");
    let e_stamp = inner
        .ladder_entries_for_test()
        .into_iter()
        .find(|&(_, hash, _)| hash == e_final)
        .map(|(_, _, stamp)| stamp)
        .expect("E stays");
    assert!(
        table < written && written < e_stamp,
        "{table} {written} {e_stamp}"
    );
    assert_eq!(inner.gdn_in_use_for_test(0), None);
    Ok(())
}

/// At the cap a rung on the prompt's tokens but cached under `keys` and
/// `salt` (another chain) is no replay point of the prompt: the prompt's
/// claim drops it and the prompt's four blocks fit.
fn cap_prompt_grow_drops_a_rung_on_another_chain(
    seed: u64,
    keys: impl Fn(&[u32], u32) -> Vec<Vec<u64>>,
    salt: u64,
) -> Result<()> {
    let mut inner = fixture(seed);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let stale = chain_tokens(30, 3, block_size);
    let stale_keys = keys(&stale, block_size);
    let hashes = cache_keyed_chain(&inner, &stale, &stale_keys, salt);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, stale.clone(), hashes, stamp);
    let mut prompt = stale;
    prompt.extend(chain_tokens(31, 1, block_size));
    let prompt_keys = per_block_keys(&prompt, block_size);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .begin_request(1)
        .map_err(Error::from_reason)?;
    clear_grow_reclaim_log();
    let mut runs = 0;
    let keep = Keep {
        tokens: &prompt,
        keys: Some((&prompt_keys, 0)),
    };
    claimed(
        &mut inner,
        keep,
        reserve_step(&mut runs, 1, prompt.len() as u32),
    )?;
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert!(inner.ladder_entries_for_test().is_empty());
    Ok(())
}

/// A rung on another cache salt.
/// Mutation: restore the token test for ladder entries in
/// `KeptPoints::keeps` → red (the rung stays, the grow log is empty).
#[test]
fn cap_prompt_grow_drops_a_rung_on_another_salt() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    cap_prompt_grow_drops_a_rung_on_another_chain(0xDFA5_1818, per_block_keys, 7)
}

/// A rung on other image keys.
/// Mutation: restore the token test for ladder entries in
/// `KeptPoints::keeps` → red (the rung stays, the grow log is empty).
#[test]
fn cap_prompt_grow_drops_a_rung_on_other_image_keys() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    cap_prompt_grow_drops_a_rung_on_another_chain(
        0xDFA5_1819,
        |tokens, block_size| vec![vec![0x1A6E]; tokens.len() / block_size as usize],
        0,
    )
}

/// At the cap a rung whose tip is a held tail registered under two
/// released blocks (a verified-duplicate prefix) is a unit: dropping it
/// lets those blocks go.
/// Mutation: keep a mark in `tip_marks` only when the mark itself is
/// cache-only → red (`Err` "could not reserve", nothing taken).
#[test]
fn cap_reclaim_takes_a_rung_over_a_held_tail() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1820);
    install_pool(&mut inner, 5, 5, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let chain = chain_tokens(16, 3, block_size);
    cache_chain(&inner, &chain[..(2 * block_size) as usize]);
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.begin_request(9).map_err(Error::from_reason)?;
        adapter
            .allocate_suffix_blocks_for(9, 3 * block_size)
            .map_err(Error::from_reason)?;
        adapter.record_tokens(&chain).map_err(Error::from_reason)?;
        adapter
            .register_full_blocks_for_reuse(&[], 0)
            .map_err(Error::from_reason)?;
    }
    let hashes = mlx_paged_attn::chain_hashes(&chain, block_size, &[], 0);
    {
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        let allocator = adapter.allocator().lock().expect("allocator");
        assert_eq!(allocator.cache_only_count(hashes[..2].iter().copied()), 2);
        assert!(!allocator.is_cache_only(hashes[2]));
    }
    push_rung(&mut inner, chain, hashes, 10);
    let blocks = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .block_telemetry()
        .map_err(Error::from_reason)?;
    assert_eq!((blocks.free_blocks, blocks.evictable_blocks), (0, 0));
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .begin_request(1)
        .map_err(Error::from_reason)?;
    clear_grow_reclaim_log();
    let mut runs = 0;
    paged_step_claimed(&mut inner, reserve_step(&mut runs, 1, block_size))?;
    assert_eq!(runs, 2);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert!(inner.ladder_entries_for_test().is_empty());
    Ok(())
}

/// A scheduled turn's close ends its replay point's use: rung E leaves
/// InUse and becomes the newest rung, above rung F.
/// Mutation: drop the `end_gdn_use` call in dense `scheduled_turn_closed`
/// → red (E stays InUse).
#[test]
fn scheduled_turn_close_ends_the_replay_points_use() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "scheduled_turn_close_ends_the_replay_points_use",
    ) {
        return Ok(());
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    let (mut inner, e_final, f_final, _) = restored_from_rung(0xDFA5_1815, false)?;
    let e_key = rung_key(&inner, 2 * pool_block_size(&inner), e_final);
    assert_eq!(inner.gdn_in_use_for_test(0), Some(&e_key));
    HybridSchedulerBackend::scheduled_turn_closed(&mut inner, 0);
    assert_eq!(inner.gdn_in_use_for_test(0), None);
    let stamp_of = |hash: u64| {
        inner
            .ladder_entries_for_test()
            .into_iter()
            .find(|&(_, final_hash, _)| final_hash == hash)
            .map(|(_, _, stamp)| stamp)
            .expect("the rung stays")
    };
    assert!(stamp_of(e_final) > stamp_of(f_final));
    Ok(())
}

struct NullSink;
impl crate::engine::backend::ChunkSink for NullSink {
    fn send(&self, _chunk: Result<crate::engine::types::ChatStreamChunk>) {}
}

/// The serial sync and stream turns at the cap, on a pool whose window
/// leaves the prompt four blocks past the recurrent rows: their prepare is
/// refused
/// until the take order drops a ladder rung the prompt does not share; the
/// prompt grow before it counts the live table as the prompt's and takes
/// nothing, so the prepare's own reclaim is the only one on the path.
/// Mutations: drop the `claimed` wrapper on the prepare in
/// `paged_turn_sync_core` → red on `sync` (`Err`); in
/// `paged_turn_stream_core` → red on `stream` (`Err`).
#[test]
fn serial_prepare_at_the_cap_frees_a_ladder_chain() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "serial_prepare_at_the_cap_frees_a_ladder_chain",
    ) {
        return Ok(());
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let rows = {
        let probe = cap_with_a_shared_live_table(0xDFA5_1816)?;
        let bytes_per_block = probe
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .bytes_per_block()
            .map_err(Error::from_reason)?;
        probe
            .config
            .recurrent_state_bytes()
            .div_ceil(bytes_per_block) as u32
    };
    for entry in ["sync", "stream"] {
        let mut inner = cap_with_a_shared_live_table_of(0xDFA5_1816, rows + 4)?;
        let block_size = pool_block_size(&inner);
        let mut prompt = chain_tokens(23, 1, block_size);
        prompt.extend([2, 3, 4, 5]);
        let config = ar_config();
        let params = ChatBackend::resolve_params(&inner, &config);
        let thinking = ChatBackend::thinking_setup(&inner, &config);
        clear_grow_reclaim_log();
        if entry == "sync" {
            inner.paged_turn_sync_core(
                prompt,
                tokenizer.clone(),
                u32::MAX,
                params,
                false,
                thinking,
            )?;
        } else {
            let sink = NullSink;
            inner.paged_turn_stream_core(
                prompt,
                tokenizer.clone(),
                u32::MAX,
                params,
                false,
                &super::paged_backend::StreamSender(&sink),
                &std::sync::atomic::AtomicBool::new(false),
                thinking,
            )?;
        }
        assert_eq!(
            grow_reclaim_log().first(),
            Some(&GrowReclaim::Ladder),
            "{entry}"
        );
    }
    Ok(())
}

/// The dense VLM prepare hits a rung's chain whose GDN state cannot
/// restore, so it restarts cold at the cap: the restart keeps nothing, so
/// the prompt's own (older) rung goes and the other stays. The candidate
/// prepare fits in the free block, so the restart's reclaim is the only one
/// on the path.
/// Mutation: drop the `claimed` wrapper on the cold restart in
/// `prepare_dense_vlm_paged_prefix` → red (`Err`).
#[test]
fn vlm_cold_restart_at_the_cap_frees_a_ladder_chain() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1817);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let kept = chain_tokens(24, 1, block_size);
    let kept_hashes = cache_chain(&inner, &kept);
    let older = inner.cache_recency.next();
    push_rung(&mut inner, kept.clone(), kept_hashes, older);
    let other = chain_tokens(25, 2, block_size);
    let other_hashes = cache_chain(&inner, &other);
    let newer = inner.cache_recency.next();
    push_rung(&mut inner, other.clone(), other_hashes, newer);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .reset_for_new_request(0)
        .map_err(Error::from_reason)?;
    let mut prompt = kept.clone();
    prompt.extend(chain_tokens(26, 1, block_size));
    let keys = per_block_keys(&prompt, block_size);
    clear_grow_reclaim_log();
    let resolution = inner.prepare_dense_vlm_paged_prefix(
        &prompt,
        prompt.len() as u32,
        block_size,
        &keys,
        true,
        true,
        77,
        0,
    )?;
    assert_eq!(resolution.candidate_cached_prefix_len as usize, kept.len());
    assert!(resolution.downgraded_to_cold);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    let left: Vec<(u32, u64, u64)> = inner.ladder_entries_for_test();
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].0 as usize, other.len());
    Ok(())
}

/// A pool of 3 blocks holding a cached 2-block chain whose first block
/// carries the only ladder rung; the prompt is that chain plus one block.
fn vlm_rung_on_the_prompts_chain(seed: u64) -> Result<(Qwen35Inner, Vec<u32>)> {
    let mut inner = fixture(seed);
    install_pool(&mut inner, 3, 3, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let chain = chain_tokens(24, 2, block_size);
    let hashes = cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    push_rung(
        &mut inner,
        chain[..block_size as usize].to_vec(),
        hashes[..1].to_vec(),
        stamp,
    );
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .reset_for_new_request(0)
        .map_err(Error::from_reason)?;
    let mut prompt = chain;
    prompt.extend(chain_tokens(26, 1, block_size));
    Ok((inner, prompt))
}

/// The cold restart restores no replay point, so it takes the prompt's
/// own rung: the candidate hits the 2-block chain past the rung, and the
/// restart needs the whole prompt fresh, which only the rung's block can
/// complete.
/// Mutation: pass the candidate's `keep` to the cold-restart `claimed`
/// in `prepare_dense_vlm_paged_prefix` → red (`Err`
/// context_length_exceeded).
#[test]
fn vlm_cold_restart_takes_the_prompts_own_rung() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let (mut inner, prompt) = vlm_rung_on_the_prompts_chain(0xDFA5_1818)?;
    let block_size = pool_block_size(&inner);
    let keys = per_block_keys(&prompt, block_size);
    clear_grow_reclaim_log();
    let resolution = inner.prepare_dense_vlm_paged_prefix(
        &prompt,
        prompt.len() as u32,
        block_size,
        &keys,
        true,
        true,
        77,
        0,
    )?;
    assert!(resolution.downgraded_to_cold);
    assert_eq!(resolution.candidate_cached_prefix_len, 2 * block_size);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert!(inner.ladder_entries_for_test().is_empty());
    Ok(())
}

/// A VLM prepare that skips the lookup keeps nothing: the pool ensure and
/// the candidate prepare take the prompt's rung for the whole fresh prompt.
/// Mutation: keep the prompt's `Keep` when `!reuse_cache` in
/// `prepare_dense_vlm_paged_prefix` → red (`Err`
/// context_length_exceeded).
#[test]
fn vlm_prepare_without_reuse_takes_the_prompts_own_rung() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let (mut inner, prompt) = vlm_rung_on_the_prompts_chain(0xDFA5_1819)?;
    let block_size = pool_block_size(&inner);
    let keys = per_block_keys(&prompt, block_size);
    clear_grow_reclaim_log();
    let resolution = inner.prepare_dense_vlm_paged_prefix(
        &prompt,
        prompt.len() as u32,
        block_size,
        &keys,
        false,
        true,
        77,
        0,
    )?;
    assert_eq!(resolution.candidate_cached_prefix_len, 0);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert!(inner.ladder_entries_for_test().is_empty());
    Ok(())
}

/// A serial turn that released its request still ends its replay point's
/// use at the history save, for `reuse_cache=false` and for a reuse turn
/// whose reconcile failed: rung E leaves InUse, becomes the newest rung,
/// and is a unit the next turn's reclaim can take.
/// Mutations: drop the `end_gdn_use` call in the `!reuse_cache` branch of
/// dense `save_paged_history` → red on `reuse false`; drop the one after
/// the history write → red on `reuse true`.
#[test]
fn a_released_serial_turn_ends_the_replay_points_use() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_released_serial_turn_ends_the_replay_points_use",
    ) {
        return Ok(());
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    for (seed, reuse) in [(0xDFA5_181A, false), (0xDFA5_181B, true)] {
        let (mut inner, e_final, f_final, prompt) = restored_from_rung(seed, false)?;
        let e_key = rung_key(&inner, 2 * pool_block_size(&inner), e_final);
        assert_eq!(inner.gdn_in_use_for_test(0), Some(&e_key), "reuse {reuse}");
        {
            crate::engine::backend::PagedBackend::finalize_paged_turn(&mut inner, false, 0);
            crate::engine::backend::PagedBackend::save_paged_history(
                &mut inner,
                &prompt,
                &[],
                false,
                reuse,
            )?;
        }
        assert_eq!(inner.gdn_in_use_for_test(0), None, "reuse {reuse}");
        let stamp_of = |inner: &Qwen35Inner, hash: u64| {
            inner
                .ladder_entries_for_test()
                .into_iter()
                .find(|&(_, final_hash, _)| final_hash == hash)
                .map(|(_, _, stamp)| stamp)
                .expect("the rung stays")
        };
        assert!(
            stamp_of(&inner, e_final) > stamp_of(&inner, f_final),
            "reuse {reuse}"
        );
        let f_key = rung_key(&inner, pool_block_size(&inner), f_final);
        let skip: std::collections::HashSet<UnitKey> = [f_key, UnitKey::History].into();
        let next = inner
            .gdn_state_units(Keep::default(), &Default::default())
            .into_iter()
            .filter(|unit| !unit.in_use && !skip.contains(&unit.key))
            .min_by_key(|unit| unit.stamp)
            .map(|unit| unit.key);
        assert_eq!(next, Some(e_key), "reuse {reuse}");
    }
    Ok(())
}

/// An AR turn's 6-block live table under a ladder rung on an 8-block pool,
/// then a cold DFlash2 turn on a 7-block prompt in 2-block chunks: the
/// prime releases the table into the cache, marked, so the second chunk is
/// refused until the take order drops the rung (Splash `Engine::growLane`
/// books each chunk through `Engine::allocate`). The turn's own states,
/// published as its chunks reach them, mark the old table's blocks under
/// their hashes: the third chunk takes the checkpoint at 64, the last one
/// the replay point at 96 (Splash takes a state in use last).
/// Mutation R1 (a DFlash2 prefill chunk books with a plain `record_tokens`)
/// → red (`Err` "could not reserve").
#[test]
fn cold_dflash2_prefill_reclaims_a_marked_live_chain() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "cold_dflash2_prefill_reclaims_a_marked_live_chain",
    ) {
        return Ok(());
    }
    use crate::engine::backend::PagedPrefix;
    use crate::engine::dspark_turn::PagedDsparkBackend;
    use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
    use crate::stream::{Stream, StreamContext};
    if !metal_is_default_device() {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let mut inner = fixture(0xDFA5_1830);
    install_pool(&mut inner, 8, 8, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let live = chain_tokens(40, 6, block_size);
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter
            .prepare_turn(0, &live, live.len() as u32, true, &[], 0, false)
            .map_err(Error::from_reason)?;
        adapter.record_tokens(&live).map_err(Error::from_reason)?;
        adapter
            .register_full_blocks_for_reuse(&[], 0)
            .map_err(Error::from_reason)?;
    }
    let hashes = mlx_paged_attn::chain_hashes(&live, block_size, &[], 0);
    let rung_tip = *hashes.last().expect("a chain");
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, live.clone(), hashes, stamp);
    let mut prompt = live;
    prompt.extend(chain_tokens(41, 1, block_size));
    clear_grow_reclaim_log();
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(2 * block_size as i32)));
    let result = (|| -> Result<()> {
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 0);
        inner
            .paged_prefill_with_draft_state(&prompt, &prefix, stream)?
            .eval();
        Ok(())
    })();
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
    result?;
    let depths: Vec<usize> = inner
        .dflash2_states
        .store
        .entries()
        .map(|entry| entry.depth())
        .collect();
    assert_eq!(
        (grow_reclaim_log(), depths),
        (
            vec![GrowReclaim::Ladder, GrowReclaim::Store, GrowReclaim::Store],
            Vec::new()
        )
    );
    assert!(
        inner
            .ladder_entries_for_test()
            .iter()
            .all(|&(_, tip, used)| (tip, used) != (rung_tip, stamp)),
        "the rung over the released table went"
    );
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    assert_eq!(adapter.current_token_count() as usize, prompt.len());
    Ok(())
}

/// A tiny tokenizer whose chat template ends thinking with `t13</think>`
/// (`</think>` = 15, not special).
fn closing_tokenizer() -> Arc<Qwen3Tokenizer> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "qwen35_closing_tokenizer_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut vocab: serde_json::Map<String, serde_json::Value> = (0..15)
        .map(|i| (format!("t{i}"), serde_json::json!(i)))
        .collect();
    vocab.insert("</think>".into(), serde_json::json!(15));
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [{
            "id": 15, "content": "</think>", "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": false,
        }],
        "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "t0" },
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&json).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("chat_template.jinja"),
        "{%- for m in messages %}{%- if m.reasoning_content %}\
         {{- m.reasoning_content + 't13</think>' }}{%- endif %}{{- m.content }}{%- endfor %}",
    )
    .unwrap();
    let tok = Qwen3Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Arc::new(tok)
}

struct LastChunk(std::sync::Mutex<Option<crate::engine::types::ChatStreamChunk>>);
impl crate::engine::backend::ChunkSink for LastChunk {
    fn send(&self, chunk: Result<crate::engine::types::ChatStreamChunk>) {
        if let Ok(chunk) = chunk
            && chunk.done
        {
            *self.0.lock().unwrap() = Some(chunk);
        }
    }
}

/// The dense whole-turn cores (sync and stream) end a budget cut with the
/// template's whole close, one token per step.
/// Mutation M12: the cores force `</think>` alone → red on both.
#[test]
fn legacy_dense_cores_end_thinking_with_the_whole_close() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "legacy_dense_cores_end_thinking_with_the_whole_close",
    ) {
        return Ok(());
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = closing_tokenizer();
    assert_eq!(&*tokenizer.think_end().thinking_end, &[13, 15]);
    let config = ChatConfig {
        temperature: Some(0.0),
        max_new_tokens: Some(4),
        ..ChatConfig::default()
    };
    let thinking = ThinkingSetup {
        enabled: true,
        budget: Some(1),
    };
    for entry in ["sync", "stream"] {
        let mut inner = fixture(0xDFA5_1831);
        install_pool(&mut inner, 16, 16, passing_probe())?;
        let params = ChatBackend::resolve_params(&inner, &config);
        let (reasoning_tokens, raw_text) = if entry == "sync" {
            let result = inner.paged_turn_sync_core(
                first_prompt(3),
                tokenizer.clone(),
                u32::MAX,
                params,
                false,
                thinking,
            )?;
            (result.reasoning_tokens, result.raw_text)
        } else {
            let sink = LastChunk(std::sync::Mutex::new(None));
            inner.paged_turn_stream_core(
                first_prompt(3),
                tokenizer.clone(),
                u32::MAX,
                params,
                false,
                &super::paged_backend::StreamSender(&sink),
                &std::sync::atomic::AtomicBool::new(false),
                thinking,
            )?;
            let last = sink.0.lock().unwrap().take().expect("final chunk");
            (
                last.reasoning_tokens.unwrap_or_default(),
                last.raw_text.unwrap_or_default(),
            )
        };
        assert_eq!(reasoning_tokens, 2, "{entry}: {raw_text:?}");
        assert!(raw_text.contains("t13 </think>"), "{entry}: {raw_text:?}");
    }
    Ok(())
}

/// A DFlash2 state of this model's geometry: a GDN cell of small arrays
/// and an empty draft window.
fn anchor(inner: &Qwen35Inner) -> crate::models::qwen3_5::dflash2_state::paged::PagedAnchor {
    use crate::models::qwen3_5::dflash2_state::paged::PagedAnchor;
    let window = inner
        .dflash2
        .as_ref()
        .expect("draft")
        .config
        .context_window()
        .expect("window");
    let mut linear = inner.fresh_dflash2_layer_caches();
    for cache in &mut linear {
        if let Qwen3_5LayerCache::Linear(gdn) = cache {
            gdn.set(
                0,
                MxArray::zeros(&[1, 3, 8], Some(crate::array::DType::BFloat16)).unwrap(),
            )
            .unwrap();
            gdn.set(
                1,
                MxArray::zeros(&[1, 2, 4, 4], Some(crate::array::DType::Float32)).unwrap(),
            )
            .unwrap();
        }
    }
    PagedAnchor { linear, window }
}

/// A parked DFlash2 state on `tokens` (whole blocks) of this model's
/// geometry, published and marked; returns its id.
fn park_state(inner: &mut Qwen35Inner, tokens: &[u32]) -> u64 {
    use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass};
    let state = anchor(inner);
    let inserted = inner
        .dflash2_states
        .store
        .insert(Lineage::text(tokens, 0), state, StateClass::Ordinary)
        .expect("a whole-block state");
    assert!(inserted.new);
    let id = inserted.id;
    inner.sync_gdn_state_marks();
    id
}

fn evictable(inner: &Qwen35Inner) -> u32 {
    inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .block_telemetry()
        .expect("telemetry")
        .evictable_blocks
}

/// A parked state keeps its chain out of `allocate` through its mark, and
/// taking it as a reclaim unit hands the chain back (Splash
/// `StateCache::evict`: the chain becomes dead leaves).
/// Mutation LK1 (the take skips the mark re-sync) → red (marked 6, nothing
/// evictable).
#[test]
fn dropping_a_parked_state_returns_its_blocks() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1901);
    install_pool(&mut inner, 8, 8, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let tokens = chain_tokens(51, 6, block_size);
    cache_chain(&inner, &tokens);
    let before = evictable(&inner);
    let id = park_state(&mut inner, &tokens);
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    assert_eq!(adapter.marked_blocks(), 6);
    assert_eq!(evictable(&inner) + 6, before);
    let taken =
        crate::models::qwen3_5::dflash2_turn::take_state_unit(&mut inner, &UnitKey::Store(id));
    assert_eq!(taken, crate::models::qwen3_5::dflash2_turn::StateTake::Took);
    assert_eq!(inner.dflash2_states.store.bytes().entries, 0);
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .marked_blocks(),
        0
    );
    assert_eq!(evictable(&inner), before);
    Ok(())
}

/// A weight update drops every parked state and its marks; a command reset
/// purges the cache and the store with it.
/// Mutation RS1 (the training clear skips the mark re-sync) → red
/// (marked 4).
#[test]
fn resets_empty_the_store_and_its_marks() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1903);
    install_pool(&mut inner, 8, 8, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let tokens = chain_tokens(54, 4, block_size);
    cache_chain(&inner, &tokens);
    park_state(&mut inner, &tokens);
    let marked = |inner: &Qwen35Inner| inner.paged_adapter.as_ref().expect("a").marked_blocks();
    assert_eq!(marked(&inner), 4);
    inner.apply_gradients_inner(Default::default(), 0.0, &Default::default())?;
    assert_eq!(inner.dflash2_states.store.bytes().entries, 0);
    assert_eq!(marked(&inner), 0);

    park_state(&mut inner, &tokens);
    ChatBackend::reset_caches(&mut inner, crate::engine::backend::ResetScope::Command)?;
    assert_eq!(inner.dflash2_states.store.bytes().entries, 0);
    assert_eq!(marked(&inner), 0);
    Ok(())
}

/// A state never outlives its block (Splash `StateCache::acquireBlock`):
/// installing one whose block left the cache fails the turn.
/// Mutation EV1 (no residency check in the install) → red (`Ok`).
#[test]
fn installing_a_state_whose_block_is_gone_fails() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1904);
    install_pool(&mut inner, 8, 8, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let tokens = chain_tokens(55, 3, block_size);
    cache_chain(&inner, &tokens);
    let id = park_state(&mut inner, &tokens);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .release_request_and_purge_prefix_cache()
        .map_err(Error::from_reason)?;
    let error = crate::models::qwen3_5::dflash2_turn::install_parked_state(&mut inner, id, false)
        .err()
        .map(|error| error.reason);
    assert!(
        error
            .as_deref()
            .is_some_and(|reason| reason.contains("outlived its target KV block")),
        "{error:?}"
    );
    Ok(())
}

/// The model's units for a ledger query: the pool's empty extents, then
/// parked states and ladder rungs in stamp order, each with its bytes and
/// the blocks its take hands back; the kept prompt's rung is no unit. A
/// rung whose arrays another holder keeps frees no bytes when taken, only
/// its block.
/// Mutations: K3 (the kept prompt is not filtered) → red (four states);
/// K3b (a take credits the unit's bytes) → red (the shared rung credits
/// its bytes).
#[test]
fn model_units_list_bytes_and_blocks_per_unit() -> Result<()> {
    use crate::engine::governor::ledger::{
        CertainReuse, Need, Reserves, Stage, TakeRights, UnitKey as LedgerKey, VictimQuery,
        VictimSource,
    };
    use crate::models::qwen3_5::dflash2_turn::ModelUnits;
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_2101);
    install_pool(&mut inner, 4, 12, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let [a, b, r, k] = [61, 62, 63, 64].map(|seed| chain_tokens(seed, 1, block_size));
    cache_chain(&inner, &a);
    cache_chain(&inner, &b);
    let r_hashes = cache_chain(&inner, &r);
    let k_hashes = cache_chain(&inner, &k);
    let a_id = park_state(&mut inner, &a);
    let b_id = park_state(&mut inner, &b);
    let r_caches = owned_caches(&inner.config);
    let r_bytes = Qwen3_5LayerCache::linear_state_bytes(&r_caches);
    let second_holder = super::state::clone_dense_linear_layer_caches(&inner.config, &r_caches)
        .expect("ready caches");
    let r_stamp = inner.cache_recency.next();
    inner.push_ladder_entry_for_test(block_size, r_hashes.clone(), r, r_caches, r_stamp);
    let k_stamp = inner.cache_recency.next();
    inner.push_ladder_entry_for_test(
        block_size,
        k_hashes,
        k.clone(),
        owned_caches(&inner.config),
        k_stamp,
    );
    let extent_bytes = {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.set_extent_growth(4);
        assert!(adapter.try_grow_pool(8));
        8 * adapter.bytes_per_block().map_err(Error::from_reason)?
    };
    let entry_bytes = |id: u64| {
        inner
            .dflash2_states
            .store
            .entries()
            .find(|entry| entry.id() == id)
            .map(|entry| entry.state_bytes())
            .expect("parked")
    };
    let (a_bytes, b_bytes) = (entry_bytes(a_id), entry_bytes(b_id));
    let mut prompt = k;
    prompt.push(1);
    let keep = Keep {
        tokens: &prompt,
        keys: None,
    };
    let r_key = LedgerKey::Ladder {
        owner_id: inner.active_cache_owner_id.clone(),
        prefix_len: block_size,
        final_block_hash: r_hashes[0],
    };
    let query = VictimQuery {
        rights: TakeRights {
            units: true,
            in_use: true,
            certain_reuse: CertainReuse::All,
            reserves: Reserves::All,
            ..TakeRights::default()
        },
        need: Need {
            bytes: 1,
            ..Need::default()
        },
    };
    let mut units = ModelUnits {
        target: &mut inner,
        keep,
    };
    let listed: Vec<(LedgerKey, u64, u32, Stage)> = units
        .units(&query)
        .into_iter()
        .map(|unit| (unit.key, unit.bytes, unit.pages, unit.stage))
        .collect();
    assert_eq!(
        listed,
        vec![
            (LedgerKey::Extent, extent_bytes, 0, Stage::FreeNow),
            (LedgerKey::Store(a_id), a_bytes, 1, Stage::FreeNow),
            (LedgerKey::Store(b_id), b_bytes, 1, Stage::FreeNow),
            (r_key.clone(), r_bytes, 1, Stage::FreeNow),
        ]
    );
    let victim = units
        .units(&query)
        .into_iter()
        .find(|unit| unit.key == r_key)
        .expect("the rung");
    let taken = units.take(&query, &victim);
    assert!(
        taken.now_bytes < r_bytes,
        "a rung another holder keeps credited {} of {r_bytes} bytes",
        taken.now_bytes
    );
    assert_eq!(taken.pages, 1);
    drop(second_holder);
    Ok(())
}

/// Splash `Cache::walkFor`: a waiting prompt's rung R is kept from a
/// Start claim and taken by an InService one (certain reuse); the newest
/// parked state, the resume point, is kept from the host's deficit pass
/// and taken by a Start.
/// Mutation Q1 (wanted chains ignored) → red (the Start lists R).
#[test]
fn a_start_keeps_queued_chains_an_in_service_claim_takes_them() -> Result<()> {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, lane, model, stats};
    use crate::engine::governor::host::Pressure;
    use crate::engine::governor::ledger::{
        ClaimClass, Need, NeedTiming, ReachClass, VictimQuery, VictimSource, WantedChains,
        take_rights,
    };
    use crate::models::qwen3_5::dflash2_turn::ModelUnits;
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_3701);
    install_pool(&mut inner, 4, 12, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let [a, b, r] = [71, 72, 73].map(|seed| chain_tokens(seed, 1, block_size));
    cache_chain(&inner, &a);
    cache_chain(&inner, &b);
    let r_hashes = cache_chain(&inner, &r);
    let a_id = park_state(&mut inner, &a);
    let b_id = park_state(&mut inner, &b);
    let r_stamp = inner.cache_recency.next();
    inner.push_ladder_entry_for_test(
        block_size,
        r_hashes.clone(),
        r.clone(),
        owned_caches(&inner.config),
        r_stamp,
    );
    let io = FakeIo::new(stats(0), host(32 * G, Pressure::Normal));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    let mut wanted = r.clone();
    wanted.push(1);
    governor.set_wanted_chains(7, WantedChains::new(vec![wanted]));
    let r_key = UnitKey::Ladder {
        owner_id: inner.active_cache_owner_id.clone(),
        prefix_len: block_size,
        final_block_hash: r_hashes[0],
    };
    let need = Need {
        bytes: 1,
        ..Need::default()
    };
    let mut host_deficit = lane(ClaimClass::Host, 0, 0);
    host_deficit.timing = NeedTiming::Immediate;
    let units = ModelUnits {
        target: &mut inner,
        keep: Keep::default(),
    };
    let listed = |cls_spec: crate::engine::governor::ledger::ClaimSpec| {
        let query = VictimQuery {
            rights: take_rights(&cls_spec, &need, false),
            need,
        };
        units
            .units(&query)
            .into_iter()
            .map(|unit| unit.key)
            .filter(|key| !matches!(key, UnitKey::Extent))
            .collect::<Vec<_>>()
    };
    let mut start = lane(ClaimClass::Start, 1, 0);
    start.reach = ReachClass::Ordinary;
    assert_eq!(
        listed(start),
        vec![UnitKey::Store(a_id), UnitKey::Store(b_id)],
        "a Start keeps the waiting prompt's rung and takes the resume point"
    );
    assert_eq!(
        listed(lane(ClaimClass::InService, 1, 0)),
        vec![UnitKey::Store(a_id), UnitKey::Store(b_id), r_key],
        "an InService claim takes every state"
    );
    assert_eq!(
        listed(host_deficit),
        vec![UnitKey::Store(a_id)],
        "the host's deficit pass keeps the resume point and the waiting rung"
    );
    Ok(())
}

/// A serial step at the pool's max with nothing to take ends the turn
/// `capacity_exhausted`, naming the KV target and the pages it lacked.
/// Mutation E1 (the adapter's refusal is returned unchanged) → red
/// (`context_length_exceeded:`).
#[test]
fn serial_refusal_fails_capacity_exhausted() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_2201);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    hold_blocks(&mut inner, 2, 1);
    hold_blocks(&mut inner, 1, 3);
    let mut runs = 0;
    let error = paged_step_claimed(&mut inner, reserve_step(&mut runs, 1, 4 * block_size))
        .expect_err("nothing fits");
    assert!(
        error
            .reason
            .starts_with("capacity_exhausted: could not allocate KV target: "),
        "{}",
        error.reason
    );
    assert!(
        error.reason.contains("additional_pages=1, free_pages=0"),
        "{}",
        error.reason
    );
    assert_eq!(runs, 1);
    Ok(())
}

/// A lone turn whose need exceeds every unit is counted before anything is
/// taken: it ends `capacity_exhausted` with its parked state, its ladder
/// rung and their marked chains intact.
/// Mutation B3 (take before the count covers) → red (the units are taken
/// and the step runs twice).
#[test]
fn doomed_lone_turn_keeps_the_cache() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_2501);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    hold_blocks(&mut inner, 2, 1);
    hold_blocks(&mut inner, 1, 1);
    let parked = chain_tokens(71, 1, block_size);
    cache_chain(&inner, &parked);
    park_state(&mut inner, &parked);
    let rung = chain_tokens(72, 1, block_size);
    let hashes = cache_chain(&inner, &rung);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, rung, hashes, stamp);
    clear_grow_reclaim_log();
    let mut runs = 0;
    let error = paged_step_claimed(&mut inner, reserve_step(&mut runs, 1, 4 * block_size))
        .expect_err("three blocks short, two takeable");
    assert!(
        error.reason.starts_with("capacity_exhausted: "),
        "{}",
        error.reason
    );
    assert_eq!(runs, 1);
    assert!(grow_reclaim_log().is_empty());
    assert_eq!(inner.dflash2_states.store.entries().count(), 1);
    assert_eq!(inner.ladder_entries_for_test().len(), 1);
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .marked_blocks(),
        2
    );
    Ok(())
}

/// Splash `HostRule::admits` under macOS Warning: the optional publication
/// of a ladder rung is held back and counted; the
/// capture of the prompt's replay point, in service and reaching what turns
/// use, is granted past the host since no state it may take meets it; and a
/// decode step whose pool grow the host refused is granted past it once
/// nothing in place meets it.
/// Mutation W4 (Optional judged as InService) → red (the rung is stored);
/// W5 (the replay point as Optional) → red (it is refused).
#[test]
fn warning_holds_back_publications_not_the_lane() -> Result<()> {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    use crate::engine::governor::host::Pressure;
    use crate::models::qwen3_5::dflash2_state::paged::TurnStates;
    use crate::models::qwen3_5::dflash2_state::plan::{StateBoundary, StatePlan};
    use crate::models::qwen3_5::dflash2_state::store::StateBytes;
    use crate::models::qwen3_5::dflash2_turn::reserve_capture;
    use crate::models::qwen3_5::paged_forward::CaptureGrant;
    use crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint;
    if !metal_is_default_device() {
        return Ok(());
    }
    let warning = || -> &'static Governor {
        let io = FakeIo::new(stats(0), host(32 * G, Pressure::Warning));
        Box::leak(Box::new(Governor::new(io)))
    };
    let mut inner = fixture(0xDFA5_3801);
    install_pool(&mut inner, 4, 12, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let governor = warning();
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));

    let rung = chain_tokens(91, 1, block_size);
    cache_chain(&inner, &rung);
    let caches = owned_caches(&inner.config);
    inner.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
        &rung,
        &per_block_keys(&rung, block_size),
        0,
        vec![MaterializedGdnPrefixCheckpoint {
            prefix_len: block_size,
            caches,
        }],
    );
    assert!(
        inner.ladder_entries_for_test().is_empty(),
        "the rung is held back"
    );

    let row = block_size;
    inner.dflash2_states.turn = Some(TurnStates {
        plan: StatePlan {
            boundaries: vec![StateBoundary {
                tokens: row,
                disposable: false,
            }],
            prompt_replay: row,
        },
        cursor: 0,
        passed: Vec::new(),
        anchor_bytes: anchor(&inner).state_bytes(),
        latest_checkpoint: None,
        cache_salt: 0,
        junction: 0,
        prompt_tokens: row + 1,
        reuse: true,
    });
    assert!(
        matches!(
            reserve_capture(&mut inner, row)?,
            CaptureGrant::Reserved(Some(_), _)
        ),
        "the replay point's capture is granted"
    );
    inner.dflash2_states.turn = None;
    assert_eq!(governor.publications_refused(7), 1);

    let mut lane = fixture(0xDFA5_3802);
    install_pool(&mut lane, 4, 12, passing_probe())?;
    let governor = warning();
    lane.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    {
        let adapter = lane.paged_adapter.as_mut().expect("adapter");
        adapter.set_admission(Arc::new(governor));
        adapter.set_extent_growth(4);
    }
    hold_blocks(&mut lane, 2, 3);
    hold_blocks(&mut lane, 1, 1);
    let mut runs = 0;
    paged_step_claimed(&mut lane, reserve_step(&mut runs, 1, 2 * block_size))?;
    assert_eq!(runs, 2, "refused by the host, then granted past it");
    assert!(
        lane.paged_adapter
            .as_ref()
            .expect("adapter")
            .block_telemetry()
            .expect("telemetry")
            .total_blocks
            > 4
    );
    Ok(())
}

/// Splash: an idle engine gives back under pressure. With no request and
/// macOS critical, the monitor's wake reaches the idle model thread, whose
/// control pass takes the parked state and the pool's empty extent.
/// Mutation K2 (the control pass is skipped while idle) → red (both stay).
#[test]
fn idle_thread_gives_back_under_critical() -> Result<()> {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    use crate::engine::governor::host::Pressure;
    use crate::engine::hybrid_scheduler::HybridSchedulerState;
    use crate::model_thread::LoopControl;
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_3901);
    install_pool(&mut inner, 4, 12, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let parked = chain_tokens(94, 1, block_size);
    cache_chain(&inner, &parked);
    park_state(&mut inner, &parked);
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.set_extent_growth(4);
        assert!(adapter.try_grow_pool(8));
        assert!(adapter.free_extent_bytes(false) > 0);
    }
    let io = FakeIo::new(stats(0), host(32 * G, Pressure::Critical));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    let mut state = HybridSchedulerState::new(inner)?;
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    sender
        .send(Qwen35Cmd::MemoryControl)
        .map_err(|_| Error::from_reason("send"))?;
    assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
    assert_eq!(state.inner.dflash2_states.store.entries().count(), 0);
    assert_eq!(
        state
            .inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .free_extent_bytes(false),
        0
    );
    Ok(())
}

/// Splash `Cache::takeReclaim`: a device need takes the dead leaves and
/// releases the extent they empty. A pool grown by an extent whose eight
/// blocks are two cached chains nothing holds (no free slot, so no empty
/// extent) gives that extent back to a critical host pass: the oldest chain
/// goes and the newer one moves down. The chain the claim's prompt walks
/// stays: the newer one goes instead.
/// Mutation R3 (a dead-leaves take evicts nothing) → red (eight blocks).
#[test]
fn critical_host_pass_evicts_dead_leaves_to_cut_an_extent() -> Result<()> {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    use crate::engine::governor::host::Pressure;
    use crate::models::qwen3_5::dflash2_turn::ModelUnits;
    if !metal_is_default_device() {
        return Ok(());
    }
    for keep_old in [false, true] {
        let mut inner = fixture(0xDFA5_3911);
        install_pool(&mut inner, 4, 12, passing_probe())?;
        let block_size = pool_block_size(&inner);
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .set_extent_growth(4);
        let old = chain_tokens(95, 4, block_size);
        let new = chain_tokens(96, 4, block_size);
        let old_hashes = cache_chain(&inner, &old);
        assert!(
            inner
                .paged_adapter
                .as_mut()
                .expect("adapter")
                .try_grow_pool(4)
        );
        let new_hashes = cache_chain(&inner, &new);
        assert_eq!(
            inner
                .paged_adapter
                .as_ref()
                .expect("adapter")
                .allocator()
                .lock()
                .unwrap()
                .num_blocks(),
            8
        );
        assert_eq!(
            inner
                .paged_adapter
                .as_ref()
                .expect("adapter")
                .free_extent_bytes(false),
            0,
            "no empty extent"
        );
        let io = FakeIo::new(stats(0), host(32 * G, Pressure::Critical));
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        let registration = governor.register_model(model(16 * G, 0)).expect("register");
        let model_id = registration.model_id();
        inner.memory_registration = Some(registration);
        let mut prompt = old.clone();
        prompt.push(1);
        let mut units = ModelUnits {
            target: &mut inner,
            keep: Keep {
                tokens: if keep_old { &prompt } else { &[] },
                keys: None,
            },
        };
        governor.control_pass(model_id, &mut [&mut units]);
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert_eq!(
            adapter.allocator().lock().unwrap().num_blocks(),
            4,
            "keep_old={keep_old}: one extent cut"
        );
        let (kept, gone) = if keep_old {
            (&old_hashes, &new_hashes)
        } else {
            (&new_hashes, &old_hashes)
        };
        assert!(
            kept.iter().all(|&hash| adapter.prefix_cached(hash)),
            "keep_old={keep_old}"
        );
        assert!(
            gone.iter().all(|&hash| !adapter.prefix_cached(hash)),
            "keep_old={keep_old}"
        );
    }
    Ok(())
}

/// Splash `Cache::WalkCursor::Phase` (`Checkpoints`, `DeadLeaves`,
/// `Ordinary`): a bytes need lists a stored checkpoint before the pool's
/// dead leaves, and the dead leaves before an ordinary state.
/// Mutation DL1 (the dead leaves go ahead of the checkpoints) → red.
#[test]
fn dead_leaves_go_after_checkpoints_before_ordinary_states() -> Result<()> {
    use crate::engine::governor::ledger::{CertainReuse, Need, Reserves, TakeRights, VictimQuery};
    use crate::models::qwen3_5::dflash2_state::paged::PagedAnchor;
    use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass};
    use crate::models::qwen3_5::dflash2_turn::model_units;
    use crate::models::qwen3_5::draft_context::DraftContextWindow;
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_3912);
    install_pool(&mut inner, 4, 12, passing_probe())?;
    let block_size = pool_block_size(&inner);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .set_extent_growth(4);
    cache_chain(&inner, &chain_tokens(95, 4, block_size));
    assert!(
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .try_grow_pool(4)
    );
    cache_chain(&inner, &chain_tokens(96, 4, block_size));
    let anchor = || PagedAnchor {
        linear: Vec::new(),
        window: DraftContextWindow::new(64, 8).expect("window"),
    };
    let (ordinary, checkpoint) = (
        chain_tokens(97, 1, block_size),
        chain_tokens(98, 1, block_size),
    );
    let store = &mut inner.dflash2_states.store;
    let o = store
        .insert(Lineage::text(&ordinary, 0), anchor(), StateClass::Ordinary)
        .map_err(|error| Error::from_reason(format!("{error:?}")))?
        .id;
    let c = store
        .insert(
            Lineage::text(&checkpoint, 0),
            anchor(),
            StateClass::Checkpoint,
        )
        .map_err(|error| Error::from_reason(format!("{error:?}")))?
        .id;
    let query = VictimQuery {
        rights: TakeRights {
            units: true,
            certain_reuse: CertainReuse::All,
            reserves: Reserves::All,
            ..TakeRights::default()
        },
        need: Need {
            bytes: 1,
            ..Need::default()
        },
    };
    let keys: Vec<UnitKey> = model_units(&inner, Keep::default(), &query)
        .into_units()
        .into_iter()
        .map(|unit| unit.key)
        .filter(|key| matches!(key, UnitKey::Store(_) | UnitKey::DeadLeaves))
        .collect();
    assert_eq!(
        keys,
        vec![UnitKey::Store(c), UnitKey::DeadLeaves, UnitKey::Store(o)]
    );
    Ok(())
}

/// What a streamed turn sent: its text deltas, its done chunks' raw text.
#[derive(Default)]
pub(crate) struct StreamLog {
    pub(crate) deltas: String,
    pub(crate) done: Vec<String>,
    pub(crate) tokens: u32,
    pub(crate) finish: Option<String>,
}

/// Records a streamed turn; at its first chunk it runs `first` once.
/// `stopped` is its `ChatStreamHandle::stop` bit.
pub(crate) struct LoggingSink {
    pub(crate) log: std::sync::Mutex<StreamLog>,
    first: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>>,
    stopped: Arc<std::sync::atomic::AtomicBool>,
}

impl LoggingSink {
    pub(crate) fn new(first: Option<Box<dyn FnOnce() + Send>>) -> Self {
        Self {
            log: std::sync::Mutex::new(StreamLog::default()),
            first: std::sync::Mutex::new(first),
            stopped: Arc::default(),
        }
    }
}

impl crate::engine::backend::ChunkSink for LoggingSink {
    fn stop_requested(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn send(&self, chunk: Result<crate::engine::types::ChatStreamChunk>) {
        if let Some(first) = self.first.lock().unwrap().take() {
            first();
        }
        let Ok(chunk) = chunk else {
            return;
        };
        let mut log = self.log.lock().unwrap();
        if chunk.done {
            log.done.push(chunk.raw_text.unwrap_or_default());
            log.tokens = chunk.num_tokens.unwrap_or(0);
            log.finish = chunk.finish_reason;
        } else {
            log.deltas.push_str(&chunk.text);
        }
    }
}

const SUSPEND_MAX_NEW: u32 = 9;

fn run_streamed_turn(
    inner: &mut Qwen35Inner,
    decoder: DecoderPlan,
    tokens: &[u32],
    sink: &LoggingSink,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    run_streamed_turn_for(inner, decoder, tokens, sink, cancelled, SUSPEND_MAX_NEW)
}

pub(crate) fn run_streamed_turn_for<B: ChatBackend>(
    inner: &mut B,
    decoder: DecoderPlan,
    tokens: &[u32],
    sink: &LoggingSink,
    cancelled: &std::sync::atomic::AtomicBool,
    max_new: u32,
) -> Result<()> {
    let tokenizer = tokenizer();
    let config = ChatConfig {
        temperature: Some(0.0),
        max_new_tokens: Some(max_new as i32),
        reasoning_effort: Some("none".to_string()),
        ..ChatConfig::default()
    };
    let params = ChatBackend::resolve_params(inner, &config);
    let thinking = ChatBackend::thinking_setup(inner, &config);
    let mut args = WholeTurnArgs {
        tokens,
        tokenizer: &tokenizer,
        eos_id: u32::MAX,
        config: &config,
        params: &params,
        thinking,
        plan: TurnPlan {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            use_paged_attention: true,
            decoder,
        },
        sink: Some(sink),
        cancelled: Some(cancelled),
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };
    ChatBackend::run_paged_turn(inner, &mut args)?;
    Ok(())
}

/// A serial lane of two blocks whose prompt leaves four rows: the fifth
/// decode step needs a third block, which only a pool grow gives. The pool
/// grows through `governor`. Its GDN heads are 128 wide: a DFlash2 verify
/// runs only the deferred complete kernel.
fn suspending_lane(
    seed: u64,
    governor: &'static crate::engine::governor::governor::Governor,
) -> Result<(Qwen35Inner, Vec<u32>)> {
    use crate::engine::governor::governor::tests::{G, model};
    let mut inner = tiny_dflash_inner_with_config(seed, |config| {
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
    });
    install_pool(&mut inner, 2, 8, passing_probe())?;
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    let block_size = pool_block_size(&inner);
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.set_admission(Arc::new(governor));
        adapter.set_extent_growth(2);
    }
    let prompt: Vec<u32> = (0..2 * block_size - 4)
        .map(|i| (i * 7 + 3) % 15 + 1)
        .collect();
    Ok((inner, prompt))
}

pub(crate) fn fake_governor(
    pressure: crate::engine::governor::host::Pressure,
) -> (
    Arc<crate::engine::governor::governor::tests::FakeIo>,
    &'static crate::engine::governor::governor::Governor,
) {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, stats};
    let io = FakeIo::new(stats(0), host(32 * G, pressure));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
    (io, governor)
}

/// Splash `Engine::suspendLane` on the serial lane: a greedy streamed turn
/// whose fifth decode step the host refuses (macOS critical) vacates its
/// lane (its two blocks a cached chain nothing pins, its sequence
/// released), sends nothing more, and waits in place: the host never
/// recovers, so the turn ends `resource_timeout` at the 30 s limit; a
/// cancel during the wait ends it with the ordinary cancel error. While
/// it waits its claim counts in the host's deficit (Splash
/// `Engine::syncClaims` sets `hostRefusedWaiting`).
/// Mutation Y2 (the Yield fails the turn `capacity_exhausted`) → red; HW1
/// (a whole turn's wait syncs no claim) → red (0 bytes refused waiting).
#[test]
fn host_yield_vacates_and_waits_in_place() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("host_yield_vacates_and_waits_in_place") {
        return Ok(());
    }
    use crate::engine::governor::governor::CancelScope;
    use crate::engine::governor::governor::tests::{G, host};
    use crate::engine::governor::host::Pressure;
    use std::sync::atomic::{AtomicBool, Ordering};
    if !metal_is_default_device() {
        return Ok(());
    }
    for cancel in [false, true] {
        let (io, governor) = fake_governor(Pressure::Normal);
        let (mut inner, prompt) = suspending_lane(0xDFA5_4701, governor)?;
        let allocator = Arc::clone(inner.paged_adapter.as_ref().expect("adapter").allocator());
        let cancelled = Arc::new(AtomicBool::new(false));
        let at_first_sleep: Arc<std::sync::Mutex<Option<u32>>> = Arc::default();
        let refused_waiting: Arc<std::sync::Mutex<Option<u64>>> = Arc::default();
        {
            let seen = Arc::clone(&at_first_sleep);
            let refused = Arc::clone(&refused_waiting);
            let cancelled = Arc::clone(&cancelled);
            io.with(|state| {
                state.on_sleep = Some(Box::new(move |_| {
                    let mut seen = seen.lock().unwrap();
                    if seen.is_none() {
                        *seen = Some(allocator.lock().unwrap().num_reusable_blocks());
                        *refused.lock().unwrap() =
                            Some(governor.ledger(|ledger| ledger.host_refused_waiting_bytes()));
                    }
                    if cancel {
                        cancelled.store(true, Ordering::Relaxed);
                    }
                }));
            });
        }
        let critical = {
            let io = Arc::clone(&io);
            Box::new(move || io.with(|state| state.host = host(32 * G, Pressure::Critical)))
        };
        let sink = LoggingSink::new(Some(critical));
        let error = {
            let _scope = CancelScope::enter(Arc::clone(&cancelled));
            run_streamed_turn(
                &mut inner,
                DecoderPlan::Autoregressive,
                &prompt,
                &sink,
                &cancelled,
            )
            .expect_err("the host never recovers")
        };
        assert_eq!(
            *at_first_sleep.lock().unwrap(),
            Some(2),
            "cancel={cancel}: the lane's two blocks are a cached chain no request pins"
        );
        assert_eq!(governor.suspensions(7), 1, "cancel={cancel}");
        assert_eq!(governor.resumptions(7), 0, "cancel={cancel}");
        assert!(
            refused_waiting
                .lock()
                .unwrap()
                .is_some_and(|bytes| bytes > 0),
            "cancel={cancel}: the suspended turn's claim counts in the host's deficit"
        );
        assert_eq!(
            governor.ledger(|ledger| ledger.host_refused_waiting_bytes()),
            0,
            "cancel={cancel}: no longer once the turn ended"
        );
        let log = sink.log.lock().unwrap();
        assert!(log.done.is_empty(), "cancel={cancel}: no done chunk");
        assert!(
            !log.deltas.is_empty(),
            "cancel={cancel}: tokens before the yield"
        );
        if cancel {
            assert_eq!(
                error.reason,
                crate::engine::session::CHAT_SESSION_CANCELLED,
                "{}",
                error.reason
            );
        } else {
            assert!(
                error.reason.starts_with("resource_timeout:"),
                "{}",
                error.reason
            );
            let now = io.with(|state| state.now);
            assert!((30.0..30.2).contains(&now), "timed out at {now}");
        }
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert_eq!(adapter.request_count(), 0, "cancel={cancel}");
    }
    Ok(())
}

/// Splash `Engine::stop` on a suspended request: `settle` →
/// `finish(Stop)` takes no lane back. A stop during the serial lane's
/// suspend wait ends the turn with one done chunk, "cancelled" (the stream
/// sink reports it "stop"), on the tokens it generated; the lane stays
/// vacated (no request, no live state) and nothing is replayed. The DFlash2 lane's stepper and its pending GDN rows go with
/// the vacated caches. A stop seen once the wait is past its limit still
/// ends the turn `Stop` (Splash `Engine::settle` keeps the first end).
/// Mutation VS1 (a vacated stop reports the cancel error) → red; SW1 (the
/// wait checks its limit before the stop) → red (`resource_timeout`).
#[test]
fn a_stop_in_the_suspend_wait_completes() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_stop_in_the_suspend_wait_completes") {
        return Ok(());
    }
    use crate::engine::governor::governor::CancelScope;
    use crate::engine::governor::governor::tests::{G, host};
    use crate::engine::governor::host::Pressure;
    use std::sync::atomic::{AtomicBool, Ordering};
    if !metal_is_default_device() {
        return Ok(());
    }
    for (decoder, late) in [
        (DecoderPlan::Autoregressive, false),
        (
            DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel),
            false,
        ),
        (DecoderPlan::Autoregressive, true),
        (
            DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel),
            true,
        ),
    ] {
        let (io, governor) = fake_governor(Pressure::Normal);
        let (mut inner, prompt) = suspending_lane(0xDFA5_4901, governor)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let critical = {
            let io = Arc::clone(&io);
            Box::new(move || io.with(|state| state.host = host(32 * G, Pressure::Critical)))
        };
        let sink = LoggingSink::new(Some(critical));
        {
            let stopped = Arc::clone(&sink.stopped);
            let cancelled = Arc::clone(&cancelled);
            io.with(|state| {
                state.on_sleep = Some(Box::new(move |state| {
                    if late && !cancelled.load(Ordering::Acquire) {
                        state.now += crate::engine::governor::wait::RESOURCE_WAIT_SECONDS + 1.0;
                    }
                    stopped.store(true, Ordering::Relaxed);
                    cancelled.store(true, Ordering::Release);
                }));
            });
        }
        {
            let _scope = CancelScope::enter(Arc::clone(&cancelled));
            run_streamed_turn(&mut inner, decoder, &prompt, &sink, &cancelled)
                .unwrap_or_else(|error| panic!("{decoder:?} late={late}: {}", error.reason));
        }
        assert_eq!(governor.suspensions(7), 1, "{decoder:?}");
        assert_eq!(governor.resumptions(7), 0, "{decoder:?}");
        let log = sink.log.lock().unwrap();
        assert_eq!(log.done.len(), 1, "{decoder:?}: one done chunk");
        assert_eq!(log.finish.as_deref(), Some("cancelled"), "{decoder:?}");
        assert!(
            (1..SUSPEND_MAX_NEW).contains(&log.tokens),
            "{decoder:?}: {} tokens",
            log.tokens
        );
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert_eq!(adapter.request_count(), 0, "{decoder:?}");
        assert!(inner.caches.is_none(), "{decoder:?}: no live state");
    }
    Ok(())
}

/// Splash `Engine::stop` settles a suspended request whatever its phase,
/// its replay included. The host recovers at the wait's first sleep and the
/// stop lands after the wait's own cancel check, so the replay's chunked
/// prefill is the first to see it (its flag is the backend's turn cancel;
/// one row per chunk, so the short replay is chunked): the turn
/// still ends with one done chunk, "cancelled", on the tokens it generated.
/// Mutation R4 (a cancelled replay's error passes through) → red
/// (`prefill cancelled`).
#[test]
fn a_stop_during_the_replay_completes() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_stop_during_the_replay_completes") {
        return Ok(());
    }
    use crate::engine::governor::governor::CancelScope;
    use crate::engine::governor::governor::tests::{G, host};
    use crate::engine::governor::host::Pressure;
    use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
    use std::sync::atomic::{AtomicBool, Ordering};
    if !metal_is_default_device() {
        return Ok(());
    }
    for decoder in [
        DecoderPlan::Autoregressive,
        DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel),
    ] {
        let (io, governor) = fake_governor(Pressure::Normal);
        let (mut inner, prompt) = suspending_lane(0xDFA5_4911, governor)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let replay_cancel = Arc::new(AtomicBool::new(false));
        ChatBackend::set_turn_cancel_flag(&mut inner, Some(Arc::clone(&replay_cancel)));
        let critical = {
            let io = Arc::clone(&io);
            Box::new(move || io.with(|state| state.host = host(32 * G, Pressure::Critical)))
        };
        let sink = LoggingSink::new(Some(critical));
        {
            let stopped = Arc::clone(&sink.stopped);
            let replay_cancel = Arc::clone(&replay_cancel);
            io.with(|state| {
                state.on_sleep = Some(Box::new(move |state| {
                    state.host = host(32 * G, Pressure::Normal);
                    stopped.store(true, Ordering::Relaxed);
                    replay_cancel.store(true, Ordering::Release);
                    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(1)));
                }));
            });
        }
        {
            let _scope = CancelScope::enter(Arc::clone(&cancelled));
            let turn = run_streamed_turn(&mut inner, decoder, &prompt, &sink, &cancelled);
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
            turn.unwrap_or_else(|error| panic!("{decoder:?}: {}", error.reason));
        }
        assert_eq!(governor.suspensions(7), 1, "{decoder:?}");
        let log = sink.log.lock().unwrap();
        assert_eq!(log.done.len(), 1, "{decoder:?}: one done chunk");
        assert_eq!(log.finish.as_deref(), Some("cancelled"), "{decoder:?}");
        assert!(
            (1..SUSPEND_MAX_NEW).contains(&log.tokens),
            "{decoder:?}: {} tokens",
            log.tokens
        );
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert_eq!(adapter.request_count(), 0, "{decoder:?}");
    }
    Ok(())
}

/// Every linear layer's conv history and recurrent state, as f32, of the
/// lane the last DFlash2 turn released.
fn gdn_cells() -> Vec<Vec<f32>> {
    let caches = crate::models::qwen3_5::dflash2_turn::RELEASED_LANE
        .with(|lane| lane.borrow_mut().take())
        .expect("the turn released its live state");
    caches
        .iter()
        .filter_map(|cache| match cache {
            Qwen3_5LayerCache::Linear(arrays) => Some(arrays),
            _ => None,
        })
        .flat_map(|arrays| {
            [0, 1].map(|slot| {
                let x = arrays
                    .get(slot)
                    .expect("GDN slot")
                    .astype(crate::array::DType::Float32)
                    .expect("f32");
                x.to_float32().expect("host copy").to_vec()
            })
        })
        .collect()
}

/// The largest elementwise gap between two `gdn_cells`.
fn gdn_gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max)
}

/// Splash `admit` of a suspended request and `replayTokens`: the same turn
/// suspends at its fifth decode step, the host recovers, the lane takes its
/// KV target at once and replays its tokens, and the denied step runs
/// again: the turn streams exactly a solo greedy run's tokens, its text
/// once, and one done chunk; a DFlash2 turn's tapped replay rebuilds its
/// draft window. The resume books the KV target it was suspended with
/// before the replay.
/// Mutation Y3 (the replay drops the generated tokens) → red; V3 (the
/// resume books only its chain) → red (2 blocks booked for a 3-block
/// target).
#[test]
fn suspended_turn_replays_to_the_solo_tokens() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("suspended_turn_replays_to_the_solo_tokens") {
        return Ok(());
    }
    use crate::engine::governor::governor::tests::{G, host};
    use crate::engine::governor::host::Pressure;
    use std::sync::atomic::AtomicBool;
    if !metal_is_default_device() {
        return Ok(());
    }
    for decoder in [
        DecoderPlan::Autoregressive,
        DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel),
    ] {
        let solo = {
            let (_io, governor) = fake_governor(Pressure::Normal);
            let (mut inner, prompt) = suspending_lane(0xDFA5_4801, governor)?;
            let sink = LoggingSink::new(None);
            run_streamed_turn(&mut inner, decoder, &prompt, &sink, &AtomicBool::new(false))?;
            assert_eq!(
                governor.suspensions(7),
                0,
                "{decoder:?}: the solo run grows the pool"
            );
            std::mem::take(&mut *sink.log.lock().unwrap())
        };
        let (io, governor) = fake_governor(Pressure::Normal);
        let (mut inner, prompt) = suspending_lane(0xDFA5_4801, governor)?;
        io.with(|state| {
            state.on_sleep = Some(Box::new(|state| {
                if state.sleeps >= 3 {
                    state.host = host(32 * G, Pressure::Normal);
                }
            }));
        });
        let critical = {
            let io = Arc::clone(&io);
            Box::new(move || io.with(|state| state.host = host(32 * G, Pressure::Critical)))
        };
        let sink = LoggingSink::new(Some(critical));
        crate::models::qwen3_5::dflash2_turn::take_resume_booked();
        run_streamed_turn(&mut inner, decoder, &prompt, &sink, &AtomicBool::new(false))?;
        assert_eq!(governor.suspensions(7), 1, "{decoder:?}");
        assert_eq!(governor.resumptions(7), 1, "{decoder:?}");
        let booked = crate::models::qwen3_5::dflash2_turn::take_resume_booked();
        assert_eq!(booked.len(), 1, "{decoder:?}");
        assert!(
            booked.iter().all(|(booked, target)| booked >= target),
            "{decoder:?}: the resume books its KV target at once: {booked:?}"
        );
        let log = sink.log.lock().unwrap();
        assert_eq!(log.done.len(), 1, "{decoder:?}: one done chunk");
        assert_eq!(log.tokens, SUSPEND_MAX_NEW, "{decoder:?}");
        assert_eq!(log.done, solo.done, "{decoder:?}: the solo run's tokens");
        assert_eq!(
            log.deltas, solo.deltas,
            "{decoder:?}: the text streams once"
        );
    }
    Ok(())
}

/// Splash `NativeRuntime::executionFailed`: a GPU OOM a serial DFlash2
/// verify reports after its claim (MLX throws a failed command at the
/// accept's wait) fails the turn `metal_out_of_memory` and arms the
/// model's recovery.
/// Mutation WO1: the whole-turn boundary passes the error through → red
/// (the raw MLX text, no recovery noted).
#[test]
fn a_verify_oom_outside_the_claim_arms_recovery() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_verify_oom_outside_the_claim_arms_recovery",
    ) {
        return Ok(());
    }
    use crate::engine::governor::host::Pressure;
    use std::sync::atomic::AtomicBool;
    if !metal_is_default_device() {
        return Ok(());
    }
    let (_io, governor) = fake_governor(Pressure::Normal);
    let (mut inner, prompt) = suspending_lane(0xDFA5_4901, governor)?;
    crate::models::qwen3_5::dflash2_decode::verify_fault::set(
        "[METAL] Command buffer execution failed: Insufficient Memory \
         (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)",
    );
    let sink = LoggingSink::new(None);
    let error = run_streamed_turn(
        &mut inner,
        DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel),
        &prompt,
        &sink,
        &AtomicBool::new(false),
    )
    .expect_err("the verify fails");
    assert!(
        crate::models::qwen3_5::dflash2_decode::verify_fault::take().is_none(),
        "the verify ran"
    );
    assert!(
        error.reason.starts_with("metal_out_of_memory:"),
        "{}",
        error.reason
    );
    assert!(governor.take_gpu_oom(7), "the model's recovery is armed");
    Ok(())
}

/// A DFlash2 lane suspended at its second verify holds the first commit's
/// pending GDN rows (the deferred fold). The tapped replay feeds their
/// tokens, so the fresh cells take none of them: the turn streams the
/// solo run's tokens and ends on the solo run's GDN state (the replay's
/// prefill rounds apart, so a longer run can part at a near tie).
/// Mutation AP3 (the fresh cells keep the old pending rows) → red.
#[test]
fn a_suspended_dflash2_lane_drops_its_pending_rows() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_suspended_dflash2_lane_drops_its_pending_rows",
    ) {
        return Ok(());
    }
    use crate::engine::governor::governor::tests::{G, host};
    use crate::engine::governor::host::Pressure;
    use std::sync::atomic::AtomicBool;
    if !metal_is_default_device() {
        return Ok(());
    }
    const MAX_NEW: u32 = 16;
    let decoder = DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel);
    // Eight rows left: the first verify fills the second block.
    let lane = |governor| -> Result<(Qwen35Inner, Vec<u32>)> {
        let (inner, mut prompt) = suspending_lane(0xDFA5_4A01, governor)?;
        prompt.truncate(prompt.len() - 4);
        Ok((inner, prompt))
    };
    let (solo, solo_cells) = {
        let (_io, governor) = fake_governor(Pressure::Normal);
        let (mut inner, prompt) = lane(governor)?;
        let sink = LoggingSink::new(None);
        run_streamed_turn_for(
            &mut inner,
            decoder,
            &prompt,
            &sink,
            &AtomicBool::new(false),
            MAX_NEW,
        )?;
        assert_eq!(governor.suspensions(7), 0, "the solo run grows the pool");
        (std::mem::take(&mut *sink.log.lock().unwrap()), gdn_cells())
    };
    let (io, governor) = fake_governor(Pressure::Normal);
    let (mut inner, prompt) = lane(governor)?;
    io.with(|state| {
        state.on_sleep = Some(Box::new(|state| {
            if state.sleeps >= 3 {
                state.host = host(32 * G, Pressure::Normal);
            }
        }));
    });
    let critical = {
        let io = Arc::clone(&io);
        Box::new(move || io.with(|state| state.host = host(32 * G, Pressure::Critical)))
    };
    let sink = LoggingSink::new(Some(critical));
    run_streamed_turn_for(
        &mut inner,
        decoder,
        &prompt,
        &sink,
        &AtomicBool::new(false),
        MAX_NEW,
    )?;
    assert_eq!(governor.suspensions(7), 1);
    assert_eq!(governor.resumptions(7), 1);
    let log = sink.log.lock().unwrap();
    assert_eq!(log.tokens, MAX_NEW);
    assert_eq!(log.done, solo.done, "the solo run's tokens");
    // The replay's prefill rounds apart from the verify kernel's steps
    // (0.012 measured); a pending row folded twice moves the tokens.
    let gap = gdn_gap(&gdn_cells(), &solo_cells);
    assert!(gap < 0.05, "GDN state {gap} from the solo run's");
    Ok(())
}

/// Splash `Runtime::suspend` → `QwenStateStorage::releaseLane`: a
/// suspended DFlash2 lane's two GDN cells and pending rows are gone before
/// the wait, so the replay builds its cells beside none of them; its draft
/// ring waits in the idle lane for the replay to take.
/// Mutation R1m (the stepper keeps its lane across the claim) → red.
#[test]
fn a_suspended_dflash2_lane_drops_its_cells_and_idles_its_ring() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_suspended_dflash2_lane_drops_its_cells_and_idles_its_ring",
    ) {
        return Ok(());
    }
    use crate::engine::governor::governor::tests::{G, host};
    use crate::engine::governor::host::Pressure;
    use crate::models::qwen3_5::dflash2_decode::lane_probe;
    use std::sync::atomic::AtomicBool;
    if !metal_is_default_device() {
        return Ok(());
    }
    let decoder = DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel);
    let (io, governor) = fake_governor(Pressure::Normal);
    let (mut inner, mut prompt) = suspending_lane(0xDFA5_4A02, governor)?;
    // Eight rows left: the first verify fills the second block.
    prompt.truncate(prompt.len() - 4);
    type Held = ((usize, usize), (usize, usize));
    let in_wait: Arc<std::sync::Mutex<Option<Held>>> = Arc::default();
    {
        let seen = Arc::clone(&in_wait);
        io.with(|state| {
            state.on_sleep = Some(Box::new(move |state| {
                let mut seen = seen.lock().unwrap();
                if seen.is_none() {
                    *seen = Some(lane_probe::held());
                }
                if state.sleeps >= 3 {
                    state.host = host(32 * G, Pressure::Normal);
                }
            }));
        });
    }
    let critical = {
        let io = Arc::clone(&io);
        Box::new(move || io.with(|state| state.host = host(32 * G, Pressure::Critical)))
    };
    let sink = LoggingSink::new(Some(critical));
    run_streamed_turn_for(
        &mut inner,
        decoder,
        &prompt,
        &sink,
        &AtomicBool::new(false),
        16,
    )?;
    assert_eq!(governor.suspensions(7), 1);
    assert_eq!(governor.resumptions(7), 1);
    let ((cells, held), (ring, idle)) = in_wait.lock().unwrap().expect("the lane waited");
    assert!(cells >= 4, "{cells} cell arrays recorded");
    assert_eq!(
        held, 0,
        "{held} of the lane's {cells} cell arrays outlived the vacate"
    );
    assert!(ring > 0, "{ring} ring arrays recorded");
    assert_eq!(idle, ring, "the idle lane keeps the ring");
    Ok(())
}

/// Splash `Engine::vacateLane` on a DFlash2 turn: `discardPendingStateBoundaries`,
/// `model_.suspend`, `cache_.endRequest`, and no state published. (a) The
/// host refuses the tapped prefill's chunk after 64 of a 100-token prompt
/// (checkpoints 16..80, replay point 96): the frame unwinds, the lane is
/// vacated with no turn points and no history entry while the rolling
/// checkpoint at 64 stays; the resume restores from 64 and replays the rest
/// of the prompt with the replay point 96 ordinary, which the turn then
/// uses. (b) The host refuses a verify once the lane fed 113 tokens: the
/// vacate keeps the replay point's use, the resume restores from 96 and
/// plans a disposable point at the fed tokens' last block (Splash
/// `replayStateBoundary`). Both stream an unsuspended run's tokens. The
/// resume holds its KV target before the state is installed (Splash
/// `admit`: no restore runs while the KV is refused).
/// Mutations VC1 (the vacate stores the history entry) and RK1 (the
/// restore runs before the KV target is held) → red.
#[test]
fn vacate_of_a_dflash2_turn_publishes_nothing() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("vacate_of_a_dflash2_turn_publishes_nothing")
    {
        return Ok(());
    }
    use crate::engine::governor::governor::tests::{G, model};
    use crate::engine::governor::host::Pressure;
    use crate::models::qwen3_5::dflash2_turn::{
        DFlash2Hit, SUSPEND_LOG, SuspendRecord, YIELD_FROM,
    };
    use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
    use std::sync::atomic::AtomicBool;
    if !metal_is_default_device() {
        return Ok(());
    }
    const MAX_NEW: u32 = 32;
    let decoder = DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel);
    let prompt: Vec<u32> = (0..100).map(|i| (i * 7 + 3) % 15 + 1).collect();
    let run = |yield_from: Option<usize>| -> Result<(StreamLog, Qwen35Inner, u64, u64)> {
        let (_io, governor) = fake_governor(Pressure::Normal);
        let mut inner = tiny_dflash_inner_with_config(0xDFA5_5C01, |config| {
            config.linear_key_head_dim = 128;
            config.linear_value_head_dim = 128;
        });
        install_pool(&mut inner, 12, 12, passing_probe())?;
        inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .set_admission(Arc::new(governor));
        SUSPEND_LOG.with(|log| log.borrow_mut().clear());
        YIELD_FROM.with(|from| from.set(yield_from));
        let sink = LoggingSink::new(None);
        let ran = run_streamed_turn_for(
            &mut inner,
            decoder,
            &prompt,
            &sink,
            &AtomicBool::new(false),
            MAX_NEW,
        );
        YIELD_FROM.with(|from| from.set(None));
        ran?;
        let log = std::mem::take(&mut *sink.log.lock().unwrap());
        Ok((log, inner, governor.suspensions(7), governor.resumptions(7)))
    };
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
    let result = (|| -> Result<()> {
        let (solo, _, suspensions, _) = run(None)?;
        assert_eq!(suspensions, 0);
        assert_eq!(solo.tokens, MAX_NEW);

        let (log, inner, suspensions, resumptions) = run(Some(64))?;
        assert_eq!((suspensions, resumptions), (1, 1), "prefill");
        assert_eq!(
            SUSPEND_LOG.with(|log| log.borrow_mut().split_off(0)),
            vec![
                SuspendRecord::Vacated {
                    fed: 64,
                    states: vec![(64, true)],
                    turn: false,
                    replay_point: false,
                },
                SuspendRecord::Resumed {
                    hit: Some(DFlash2Hit::Stored(64)),
                    points: vec![(80, true), (96, false)],
                },
                SuspendRecord::Restored { depth: 64 },
            ]
        );
        assert_eq!(inner.history_stamp_for_test(), None, "no history entry");
        let entries: Vec<(usize, bool)> = inner
            .dflash2_states
            .store
            .entries()
            .map(|entry| (entry.depth(), entry.checkpoint()))
            .collect();
        assert!(entries.contains(&(96, false)), "{entries:?}");
        assert_eq!(inner.dflash2_states.counters.replay_state_publications, 1);
        assert_eq!(log.done, solo.done, "prefill: the unsuspended tokens");

        let (log, inner, suspensions, resumptions) = run(Some(113))?;
        assert_eq!((suspensions, resumptions), (1, 1), "decode");
        let records = SUSPEND_LOG.with(|log| log.borrow_mut().split_off(0));
        let [
            SuspendRecord::Vacated {
                fed,
                states,
                turn,
                replay_point,
            },
            SuspendRecord::Resumed { hit, points },
            SuspendRecord::Restored { depth: 96 },
        ] = &records[..]
        else {
            panic!("{records:?}");
        };
        assert!((113..=120).contains(fed), "fed {fed}");
        assert_eq!(states, &vec![(80, true), (96, false)]);
        assert!(
            !turn && *replay_point,
            "points gone, the replay point in use"
        );
        assert_eq!(*hit, Some(DFlash2Hit::Stored(96)));
        let end = u32::try_from((fed - 1) / 16 * 16).expect("u32");
        assert_eq!(points, &vec![(end, true)]);
        assert_eq!(inner.history_stamp_for_test(), None, "no history entry");
        assert_eq!(log.done, solo.done, "decode: the unsuspended tokens");
        Ok(())
    })();
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
    result
}

/// A lane's lookup junction counts once, when its turn starts (Splash
/// `completeAdmission` records the lookup only for a request that is not
/// resuming). A (200 tokens) registers its chain; B shares A's first 150,
/// finds the junction at 144 when it starts, and the host refuses its
/// prefill's chunk once it fed 20: the resume finds the junction again
/// below its checkpoint at 16 and does not count it.
/// Mutation JR1 (the lookup counts its junction, resumes included) → red.
#[test]
fn a_resumed_lane_counts_no_junction() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_resumed_lane_counts_no_junction") {
        return Ok(());
    }
    use crate::engine::governor::governor::tests::{G, model};
    use crate::engine::governor::host::Pressure;
    use crate::models::qwen3_5::dflash2_turn::YIELD_FROM;
    use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
    use std::sync::atomic::AtomicBool;
    if !metal_is_default_device() {
        return Ok(());
    }
    let decoder = DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel);
    let a: Vec<u32> = (0..200).map(|i| (i * 7 + 3) % 15 + 1).collect();
    let mut b = a[..150].to_vec();
    b.extend((0..30).map(|i| (i * 11 + 5) % 15 + 1));
    let (_io, governor) = fake_governor(Pressure::Normal);
    let mut inner = tiny_dflash_inner_with_config(0xDFA5_5C02, |config| {
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
    });
    install_pool(&mut inner, 40, 40, passing_probe())?;
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .set_admission(Arc::new(governor));
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
    let result = (|| -> Result<()> {
        let cancel = AtomicBool::new(false);
        run_streamed_turn_for(&mut inner, decoder, &a, &LoggingSink::new(None), &cancel, 4)?;
        assert_eq!(inner.dflash2_states.counters.lazy_junctions, 0);
        YIELD_FROM.with(|from| from.set(Some(20)));
        let ran =
            run_streamed_turn_for(&mut inner, decoder, &b, &LoggingSink::new(None), &cancel, 4);
        YIELD_FROM.with(|from| from.set(None));
        ran?;
        assert_eq!(governor.resumptions(7), 1);
        assert_eq!(inner.dflash2_states.counters.lazy_junctions, 1);
        Ok(())
    })();
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
    result
}

/// A pool at its runway whose grow macOS refuses (critical), sequence 0
/// holding both blocks.
fn refused_runway(
    seed: u64,
) -> Result<(
    Arc<crate::engine::governor::governor::tests::FakeIo>,
    &'static crate::engine::governor::governor::Governor,
    Qwen35Inner,
)> {
    use crate::engine::governor::host::Pressure;
    let (io, governor) = fake_governor(Pressure::Critical);
    let (mut inner, _) = suspending_lane(seed, governor)?;
    hold_blocks(&mut inner, 0, 2);
    Ok((io, governor, inner))
}

/// Splash `Engine::growLane` books a speculative lane's verify rows in its
/// claim, InService and Immediate: a lookahead the host refuses returns
/// false at once (the turn decodes AR, whose step's claim decides): no
/// wait, no judge, and no token event.
/// Mutation LA1 (the lookahead runs as a serial step, `paged_step_claimed`)
/// → red (it waits 30 s and fails `resource_timeout`).
#[test]
fn lookahead_refusal_decodes_ar_without_waiting() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let (io, _governor, mut inner) = refused_runway(0xDFA5_5101)?;
    let block_size = pool_block_size(&inner);
    let mut runs = 0;
    let booked = crate::models::qwen3_5::dflash2_turn::lookahead_claimed(
        &mut inner,
        reserve_step(&mut runs, 0, 3 * block_size),
    )?;
    assert!(!booked);
    assert_eq!(io.with(|state| state.sleeps), 0);
    Ok(())
}

/// The lookahead's claim frees what meets it before it refuses (Splash
/// `Engine::allocate`): at the cap a ladder rung's chain goes and the rows
/// are booked.
/// Mutation LA2 (the lookahead reserves once, nothing freed) → red (false).
#[test]
fn lookahead_takes_units_before_refusing() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_5201);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    hold_blocks(&mut inner, 1, 2);
    let chain = chain_tokens(1, 2, block_size);
    let hashes = cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, chain, hashes, stamp);
    clear_grow_reclaim_log();
    let mut runs = 0;
    let booked = crate::models::qwen3_5::dflash2_turn::lookahead_claimed(
        &mut inner,
        reserve_step(&mut runs, 1, 3 * block_size),
    )?;
    assert!(booked);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    Ok(())
}

/// Splash `Engine::admit` judges a scheduled request's KV refusal beside
/// the resident lanes: an allocation under a scheduled lane's scope
/// neither waits nor judges as a lone lane; it leaves its refusal for the
/// scheduler.
/// Mutation LJ1 (judged as a lone lane) → red (it sleeps, then fails
/// `resource_timeout`).
#[test]
fn a_scheduled_prepare_leaves_its_refusal_to_the_scheduler() -> Result<()> {
    use crate::engine::governor::governor::{Scope, take_lane_denial, whole_turn_spec};
    if !metal_is_default_device() {
        return Ok(());
    }
    let (io, governor, mut inner) = refused_runway(0xDFA5_5301)?;
    let block_size = pool_block_size(&inner);
    let claim = governor.open(whole_turn_spec(3), 7);
    let error = {
        let _lane = Scope::enter_lane(claim.claim());
        let mut runs = 0;
        claimed(
            &mut inner,
            Keep::default(),
            reserve_step(&mut runs, 0, 3 * block_size),
        )
        .expect_err("the host refuses the grow")
    };
    assert!(
        crate::engine::scheduler::is_paged_allocation_blocked(&error.reason),
        "{}",
        error.reason
    );
    assert_eq!(io.with(|state| state.sleeps), 0);
    let denial = take_lane_denial().expect("the refusal is the scheduler's");
    assert_eq!(
        denial.failure,
        Some(crate::engine::governor::governor::Failure::HostPressure)
    );
    Ok(())
}
