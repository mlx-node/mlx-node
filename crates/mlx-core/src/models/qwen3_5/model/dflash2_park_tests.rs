//! The take order a refused paged step reclaims through: ladder and history
//! entries, marked chains and image entries.

use std::sync::Arc;

use super::*;
use crate::engine::backend::{ChatBackend, WholeTurnArgs};
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
use crate::engine::reclaim::UnitKey;
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::dflash2_decode::tests::{
    metal_is_default_device, tiny_dflash_inner_with_attention_head_dim,
};
use crate::models::qwen3_5::dflash2_turn::{
    GrowReclaim, Keep, clear_grow_reclaim_log, ensure_paged_pool_for_prompt, grow_reclaim_log,
    paged_step_reclaiming, reclaiming,
};

fn fixture(seed: u64) -> Qwen35Inner {
    tiny_dflash_inner_with_attention_head_dim(seed, 32)
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

fn chain_tokens(seed: u32, blocks: u32, block_size: u32) -> Vec<u32> {
    (0..blocks * block_size)
        .map(|i| (seed * 11 + i * 7) % 15 + 1)
        .collect()
}

/// Caches `tokens` (whole blocks) as a chain nothing holds; returns its
/// block hashes.
pub(super) fn cache_chain(inner: &Qwen35Inner, tokens: &[u32]) -> Vec<u64> {
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
/// refused the same need and `MetBefore` stops the loop: `Err`).
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
    paged_step_reclaiming(&mut inner, reserve_step(&mut runs, 1, 3 * block_size))?;
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
/// Mutations: drop the `reclaiming` wrapper in dense `prime_prefix_state`
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
    let plan = reclaiming(&mut inner, keep, |inner| {
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
/// Mutations: drop the `reclaiming` wrapper in
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
/// Mutation: drop the first `sync_marks` in `free_for_kv` → red (the ladder
/// rung is dropped).
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
    paged_step_reclaiming(&mut inner, reserve_step(&mut runs, 1, block_size))?;
    assert_eq!(runs, 2);
    assert!(grow_reclaim_log().is_empty());
    assert_eq!(inner.ladder_entries_for_test().len(), 1);
    Ok(())
}

/// At the cap only the deepest of one owner's nested rungs is a unit: the
/// shallower rung lies inside its chain and would free nothing.
/// Mutation: filter on the final block being cache-only instead of
/// `tip_marks` in `oldest_gdn_state_unit` → red (the older, shallower rung
/// goes first: `[Ladder, Ladder]`).
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
    paged_step_reclaiming(&mut inner, reserve_step(&mut runs, 1, block_size))?;
    assert_eq!(runs, 2);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert_eq!(
        inner.ladder_entries_for_test(),
        vec![(2 * block_size, hashes[1], 10)]
    );
    let tips = inner.paged_adapter.as_ref().expect("adapter").tip_marks();
    assert_eq!(tips, [hashes[1]].into_iter().collect());
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
    let f_caches = restorable_caches_from(&inner, &caches);
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

/// The in-use reach counts the bytes of the rung a turn restored from; the
/// ordinary count leaves them out.
/// Mutation: ignore `in_use` in dense `gdn_state_bytes` → red (the counts
/// are equal).
#[test]
fn in_use_reach_counts_the_used_rungs_bytes() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let (inner, e_final, _, _) = restored_from_rung(0xDFA5_181C, false)?;
    let e_key = rung_key(&inner, 2 * pool_block_size(&inner), e_final);
    assert_eq!(inner.gdn_in_use_for_test(0), Some(&e_key));
    let ordinary = inner.gdn_state_bytes(Keep::default(), false);
    let reach = inner.gdn_state_bytes(Keep::default(), true);
    assert!(
        reach > ordinary,
        "in-use reach {reach}, ordinary {ordinary}"
    );
    Ok(())
}

/// The rung a running turn restored from is InUse: a reclaim passes over
/// it, though it is older than rung F, for the text and the VLM prepare.
/// Mutation: ignore `gdn_in_use` in `gdn_unit_excluded` → red (E goes
/// first).
#[test]
fn a_replay_point_in_use_is_not_a_unit() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    for (seed, vlm) in [(0xDFA5_1811, false), (0xDFA5_1812, true)] {
        let (mut inner, e_final, f_final, _) = restored_from_rung(seed, vlm)?;
        let e_key = rung_key(&inner, 2 * pool_block_size(&inner), e_final);
        assert_eq!(inner.gdn_in_use_for_test(0), Some(&e_key), "vlm {vlm}");
        assert!(
            inner.gdn_state_bytes(Keep::default(), false) > 0,
            "vlm {vlm}: the count must pass"
        );
        clear_grow_reclaim_log();
        let mut runs = 0;
        paged_step_reclaiming(&mut inner, |inner| {
            runs += 1;
            if runs == 1 {
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

/// At the cap the prompt grow's count leaves out the marked chain its own
/// prompt hits, so a need that only other rungs could not cover takes
/// nothing.
/// Mutation: drop `- marked_hit_blocks` in `ModelReclaim::reach` → red (the
/// other rung is dropped and the need is still short).
#[test]
fn cap_prompt_grow_drops_nothing_its_own_hit_inflates() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1814);
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let own = chain_tokens(20, 3, block_size);
    let own_hashes = cache_chain(&inner, &own);
    let other = chain_tokens(21, 1, block_size);
    let other_hashes = cache_chain(&inner, &other);
    let first = inner.cache_recency.next();
    push_rung(&mut inner, other, other_hashes, first);
    let second = inner.cache_recency.next();
    push_rung(&mut inner, own.clone(), own_hashes, second);
    let mut prompt = own;
    prompt.extend(chain_tokens(22, 3, block_size));
    let keys = per_block_keys(&prompt, block_size);
    clear_grow_reclaim_log();
    ensure_paged_pool_for_prompt(&mut inner, &prompt, &keys, 0);
    assert!(grow_reclaim_log().is_empty());
    assert_eq!(inner.ladder_entries_for_test().len(), 2);
    Ok(())
}

/// At the cap a rung on the prompt's tokens but cached under `keys` and
/// `salt` (another chain) is no replay point of the prompt: the prompt
/// grow drops it and the prompt's four blocks fit.
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
    clear_grow_reclaim_log();
    ensure_paged_pool_for_prompt(&mut inner, &prompt, &prompt_keys, 0);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
    assert!(inner.ladder_entries_for_test().is_empty());
    let adapter = inner.paged_adapter.as_mut().expect("adapter");
    adapter.begin_request(1).map_err(Error::from_reason)?;
    adapter
        .reserve_rows_for(1, prompt.len() as u32)
        .map_err(Error::from_reason)?;
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
    paged_step_reclaiming(&mut inner, reserve_step(&mut runs, 1, block_size))?;
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
/// Mutations: drop the `reclaiming` wrapper on the prepare in
/// `paged_turn_sync_core` → red on `sync` (`Err`); in
/// `paged_turn_stream_core` → red on `stream` (`Err`).
#[test]
fn serial_prepare_at_the_cap_frees_a_ladder_chain() -> Result<()> {
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
/// Mutation: drop the `reclaiming` wrapper on the cold restart in
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
/// Mutation: pass the candidate's `keep` to the cold-restart `reclaiming`
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
    if !metal_is_default_device() {
        return Ok(());
    }
    for (seed, reuse) in [(0xDFA5_181A, false), (0xDFA5_181B, true)] {
        let (mut inner, e_final, f_final, prompt) = restored_from_rung(seed, false)?;
        let e_key = rung_key(&inner, 2 * pool_block_size(&inner), e_final);
        assert_eq!(inner.gdn_in_use_for_test(0), Some(&e_key), "reuse {reuse}");
        {
            // The finalize bumps the global sidecar capture counter.
            let _serialized = crate::cold_tier::sidecar_counter_test_lock();
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
            .oldest_gdn_state_unit(false, &skip, Keep::default(), false)
            .map(|unit| unit.key);
        assert_eq!(next, Some(e_key), "reuse {reuse}");
    }
    Ok(())
}

/// Below the cap, the pool full under the live request and no state: the
/// prompt grow takes the image entry nothing holds, and the grow it counted
/// on then passes.
/// Mutation: `rebuilt: false` in `ensure_paged_pool_keeping` → red (the log
/// is empty, the entry stays).
#[test]
fn prompt_grow_takes_an_image_entry_below_the_cap() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1819);
    install_pool(&mut inner, 4, 8, probe_after(1))?;
    let bytes_per_block = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .bytes_per_block()
        .map_err(Error::from_reason)?;
    let block_size = pool_block_size(&inner);
    hold_blocks(&mut inner, 1, 4);
    let unheld = MxArray::from_float32(&[1.0], &[1])?;
    unheld.eval();
    inner
        .vision_cache
        .lock()
        .expect("vision cache")
        .insert_for_test(1, unheld, bytes_per_block as usize);
    let prompt = chain_tokens(40, 5, block_size);
    let keys = per_block_keys(&prompt, block_size);
    clear_grow_reclaim_log();
    ensure_paged_pool_for_prompt(&mut inner, &prompt, &keys, 0);
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Vision]);
    let cache = inner.vision_cache.lock().expect("vision cache");
    assert_eq!(cache.len(), 0, "the entry is taken");
    Ok(())
}

/// An AR turn's 6-block live table under a ladder rung on an 8-block pool,
/// then a cold DFlash2 turn on a 7-block prompt in 2-block chunks: the
/// prime releases the table into the cache, marked, so the second chunk is
/// refused until the take order drops the rung (Splash `Engine::growLane`
/// books each chunk through `Engine::allocate`).
/// Mutation R1 (a DFlash2 prefill chunk books with a plain `record_tokens`)
/// → red (`Err` "could not reserve").
#[test]
fn cold_dflash2_prefill_reclaims_a_marked_live_chain() -> Result<()> {
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
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Ladder]);
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

/// A parked DFlash2 state on `tokens` (whole blocks) of this model's
/// geometry, published and marked; returns its id.
fn park_state(inner: &mut Qwen35Inner, tokens: &[u32]) -> u64 {
    use crate::models::qwen3_5::dflash2_state::paged::PagedAnchor;
    use crate::models::qwen3_5::dflash2_state::store::Lineage;
    let window = crate::models::qwen3_5::dflash2::DFlash2ContextCache::new(
        &inner.dflash2.as_ref().expect("draft").config,
    );
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
    let (id, new) = inner
        .dflash2_states
        .store
        .insert(Lineage::text(tokens, 0), PagedAnchor { linear, window })
        .expect("a whole-block state");
    assert!(new);
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
    assert!(taken.is_some());
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

/// At the cap a parked state is a unit only when dropping it frees a
/// block: a ladder rung on the same block keeps the chain, so that state
/// is passed over for the next one.
/// Mutation CU1 (no rung filter at the cap) → red (the older state on the
/// rung's block comes first).
#[test]
fn cap_reclaim_takes_a_parked_state_that_frees_a_block() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = fixture(0xDFA5_1902);
    install_pool(&mut inner, 8, 8, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let shared = chain_tokens(52, 2, block_size);
    let alone = chain_tokens(53, 2, block_size);
    let shared_hashes = cache_chain(&inner, &shared);
    cache_chain(&inner, &alone);
    let on_rung = park_state(&mut inner, &shared);
    let free = park_state(&mut inner, &alone);
    let stamp = inner.cache_recency.next() + 100;
    push_rung(&mut inner, shared.clone(), shared_hashes, stamp);
    let unit = crate::models::qwen3_5::dflash2_turn::oldest_state_unit(
        &inner,
        true,
        &Default::default(),
        Keep::default(),
        false,
    )
    .expect("a unit");
    assert_ne!(unit.key, UnitKey::Store(on_rung));
    assert_eq!(unit.key, UnitKey::Store(free));
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
    let error = crate::models::qwen3_5::dflash2_turn::install_parked_state(&mut inner, id, 0)
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
