//! Sites that take the flat lane away from a DFlash2 state park it in the
//! model's state store; resets that end the model's history clear the store.

use std::sync::Arc;

use super::*;
use crate::engine::backend::{ChatBackend, ResetScope, WholeTurnArgs};
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
use crate::engine::reclaim::UnitKey;
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::dflash2_decode::tests::{
    array_fingerprints, metal_is_default_device, run_tiny_dflash_turn,
    tiny_dflash_inner_with_attention_head_dim,
};
use crate::models::qwen3_5::dflash2_state::store::Lineage;
use crate::models::qwen3_5::dflash2_state::trace::HitKind;
use crate::models::qwen3_5::dflash2_turn::{
    GrowReclaim, Keep, clear_grow_reclaim_log, ensure_paged_pool_for_prompt, grow_reclaim_log,
    install_flat_dflash2_end, paged_step_reclaiming, park_live_dflash2_state, reclaiming,
};

type Fingerprints = Vec<(Vec<i64>, Vec<u32>)>;

fn fixture(seed: u64) -> Qwen35Inner {
    tiny_dflash_inner_with_attention_head_dim(seed, 32)
}

fn tokenizer() -> Arc<Qwen3Tokenizer> {
    crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer()
}

fn first_prompt(seed: u32) -> Vec<u32> {
    (0..40).map(|i| (seed + i * 7) % 15 + 1).collect()
}

fn extend(history: &[u32], tail: &[u32]) -> Vec<u32> {
    history
        .iter()
        .copied()
        .chain(tail.iter().copied())
        .collect()
}

/// Every buffer of the live flat state, whole, plus its history.
fn live_state(inner: &Qwen35Inner) -> Result<(Vec<u32>, Fingerprints, Fingerprints)> {
    let mut target = Vec::new();
    for cache in inner.caches.iter().flatten() {
        cache.collect_arrays(&mut target);
    }
    let mut draft = Vec::new();
    inner
        .dflash2_context
        .as_ref()
        .expect("live draft context")
        .collect_arrays(&mut draft);
    Ok((
        inner.cached_token_history.clone(),
        array_fingerprints(target)?,
        array_fingerprints(draft)?,
    ))
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

/// The store holds exactly one end state of `history`, and a prompt that
/// extends `history` resumes on it at full depth.
fn assert_end_hit(inner: &Qwen35Inner, history: &[u32], label: &str) {
    let store = &inner.dflash2_states.store;
    assert_eq!(store.bytes().entries, 1, "{label}: one parked state");
    let next = extend(history, &[4, 5, 6]);
    let hit = store
        .lookup(Lineage::text(&next, 0), |_| true)
        .unwrap_or_else(|| panic!("{label}: the next prompt misses the store"));
    assert_eq!(
        (hit.kind, hit.depth),
        (HitKind::End, history.len()),
        "{label}"
    );
}

/// X0: the parked entry moved back in gives every buffer of the state it was
/// parked from, bit for bit. Leaves the state installed.
fn assert_parked_round_trip(
    inner: &mut Qwen35Inner,
    before: &(Vec<u32>, Fingerprints, Fingerprints),
    label: &str,
) -> Result<()> {
    let next = extend(&before.0, &[4, 5, 6]);
    let hit = inner
        .dflash2_states
        .store
        .lookup(Lineage::text(&next, 0), |_| true)
        .expect("end hit");
    let (tokens, parts) = inner
        .dflash2_states
        .store
        .take(hit.id)
        .expect("entry")
        .into_parts();
    inner.dflash2_context = None;
    inner.dflash2_turn_state = None;
    install_flat_dflash2_end(inner, tokens, parts)?;
    assert!(live_state(inner)? == *before, "{label}: X0 parked state");
    Ok(())
}

/// A paged AR turn between two DFlash2 turns of one lineage parks the first
/// turn's end state instead of dropping it, through both the generic paged
/// turn and the paged sync core. X0 for the parked buffers; X1 for the
/// DFlash2 turn resumed on them against the same two turns run back to back.
#[test]
fn paged_ar_turn_parks_the_dflash2_end_state() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let first = first_prompt(0);
    let mut reference = fixture(0xDFA5_1601);
    run_tiny_dflash_turn(&mut reference, &tokenizer, &first, true, 12)?;
    let second = extend(&reference.cached_token_history, &[4, 5, 6]);
    let reference_cached = run_tiny_dflash_turn(&mut reference, &tokenizer, &second, true, 12)?;
    let reference_state = live_state(&reference)?;

    for entry in ["run_paged_turn", "paged_turn_sync_core"] {
        let mut inner = fixture(0xDFA5_1601);
        inner.initialize_paged_adapter()?;
        run_tiny_dflash_turn(&mut inner, &tokenizer, &first, true, 12)?;
        let before = live_state(&inner)?;
        let paged = extend(&before.0, &[2, 3]);
        if entry == "run_paged_turn" {
            run_paged_ar_turn(&mut inner, &tokenizer, &paged)?;
        } else {
            let config = ar_config();
            let params = ChatBackend::resolve_params(&inner, &config);
            let thinking = ChatBackend::thinking_setup(&inner, &config);
            inner.paged_turn_sync_core(
                paged,
                tokenizer.clone(),
                u32::MAX,
                params,
                false,
                thinking,
            )?;
        }
        assert!(inner.paged_full_attn_caches_dirty, "{entry}");
        assert!(inner.dflash2_context.is_none(), "{entry}");
        assert_end_hit(&inner, &before.0, entry);
        assert_parked_round_trip(&mut inner, &before, entry)?;

        // The lane flags are the restoring turn's to clear.
        inner.paged_full_attn_caches_dirty = false;
        let resumed = run_tiny_dflash_turn(&mut inner, &tokenizer, &second, true, 12)?;
        assert_eq!(resumed.cached_tokens, before.0.len() as u32, "{entry}");
        assert_eq!(resumed.cached_tokens, reference_cached.cached_tokens);
        assert!(
            live_state(&inner)? == reference_state,
            "{entry}: X1 resumed turn"
        );
    }
    Ok(())
}

/// Swaps the model's paged pool for one of `initial` blocks that grows to
/// the same maximum, on the model's clock, with a grow probe that declines
/// until a reclaim on this thread took a unit.
fn install_runway_pool_reclaimed_by_eviction(inner: &mut Qwen35Inner, initial: u32) -> Result<()> {
    inner.paged_adapter = None;
    inner.initialize_paged_adapter()?;
    let adapter = inner.paged_adapter.take().expect("paged adapter");
    let block_size = adapter.block_size();
    let pool = adapter.layer_kv_pool();
    let (config, max, dtype) = (
        pool.config().clone(),
        pool.max_num_blocks(),
        pool.cache_dtype(),
    );
    drop(adapter);
    assert!(initial < max);
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
    clear_grow_reclaim_log();
    adapter.set_grow_headroom_probe_override(Some(Arc::new(|blocks| {
        Ok(if grow_reclaim_log().is_empty() {
            0
        } else {
            blocks
        })
    })));
    inner.paged_adapter = Some(adapter);
    Ok(())
}

/// A draft-loaded pool with no initial size starts at the runway and grows
/// by Splash extents. Mutation: drop the `set_extent_growth` call in dense
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

/// A paged AR turn right after a DFlash2 turn, with a prompt past the
/// runway pool: the turn parks the live state, and the pool grow its prompt
/// needs takes it (a grow probe reads memory the parked state holds) before
/// the prefix lookup, through both the generic paged turn and the paged sync
/// core: a compaction or a store step.
/// Mutation: skip the park in `paged_whole_turn` → red on `run_paged_turn`;
/// in `paged_turn_sync_core` → red on that entry.
#[test]
fn dflash2_to_ar_transition_reclaims_the_just_parked_state() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for entry in ["run_paged_turn", "paged_turn_sync_core"] {
        let mut inner = fixture(0xDFA5_1604);
        run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(3), true, 12)?;
        assert_eq!(inner.dflash2_states.store.bytes().entries, 0, "{entry}");
        install_runway_pool_reclaimed_by_eviction(&mut inner, 1)?;
        let paged = extend(&inner.cached_token_history, &[2, 3]);
        let block_size = inner.paged_adapter.as_ref().expect("adapter").block_size() as usize;
        assert!(
            paged.len() > block_size,
            "{entry}: the prompt outgrows the runway"
        );
        if entry == "run_paged_turn" {
            run_paged_ar_turn(&mut inner, &tokenizer, &paged)?;
        } else {
            let config = ar_config();
            let params = ChatBackend::resolve_params(&inner, &config);
            let thinking = ChatBackend::thinking_setup(&inner, &config);
            inner.paged_turn_sync_core(
                paged,
                tokenizer.clone(),
                u32::MAX,
                params,
                false,
                thinking,
            )?;
        }
        let log = grow_reclaim_log();
        assert!(
            log.first()
                .is_some_and(|taken| matches!(taken, GrowReclaim::Compaction | GrowReclaim::Store)),
            "{entry}: the just-parked state paid for the grow: {log:?}"
        );
        assert!(inner.dflash2_context.is_none(), "{entry}");
    }
    Ok(())
}

/// A vision paged turn parks the DFlash2 state before it dirties the flat
/// lane. The tiny vocabulary has no image tokens, so the turn fails at
/// image decode, after the park: X0 only.
#[test]
fn vision_paged_turn_parks_the_dflash2_end_state() -> Result<()> {
    use crate::vision::qwen::encoder::{QwenVisionConfig, QwenVisionEncoder};
    use crate::vision::qwen::processing::QwenImageProcessor;
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let mut inner = fixture(0xDFA5_1602);
    inner.initialize_paged_adapter()?;
    inner.set_vision_encoder(QwenVisionEncoder::new(QwenVisionConfig {
        hidden_size: 64,
        intermediate_size: 256,
        num_heads: 4,
        num_layers: 2,
        patch_size: 16,
        spatial_merge_size: 2,
        image_size: 256,
        out_hidden_size: inner.config.hidden_size,
    })?)?;
    inner.set_image_processor(QwenImageProcessor::new(None));
    run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(1), true, 12)?;
    let before = live_state(&inner)?;

    let config = ar_config();
    let params = ChatBackend::resolve_params(&inner, &config);
    let thinking = ChatBackend::thinking_setup(&inner, &config);
    let turn = inner.vision_paged_turn_sync_core(
        extend(&before.0, &[2, 3]),
        &[vec![0u8; 8]],
        tokenizer.clone(),
        u32::MAX,
        params,
        false,
        thinking,
    );
    assert!(turn.is_err(), "the undecodable image fails the turn");
    assert!(inner.paged_full_attn_caches_dirty);
    assert_end_hit(&inner, &before.0, "vision");
    assert_parked_round_trip(&mut inner, &before, "vision")
}

/// A prefix-miss reset parks the state and keeps the history this scope
/// preserves; the first lineage then resumes on the parked end state (X0).
#[test]
fn prefix_miss_reset_parks_the_dflash2_end_state() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let mut inner = fixture(0xDFA5_1603);
    run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(2), true, 12)?;
    let before = live_state(&inner)?;
    ChatBackend::reset_caches(&mut inner, ResetScope::PrefixMiss)?;
    assert_eq!(inner.cached_token_history, before.0);
    assert!(inner.dflash2_context.is_none());
    assert!(inner.caches.is_some(), "fresh caches for the missing turn");
    assert_end_hit(&inner, &before.0, "prefix miss");
    assert_parked_round_trip(&mut inner, &before, "prefix miss")
}

/// A command reset, a full cache reset (the generate, calibrate and eval
/// bodies) and a weight update each empty the store.
#[test]
fn command_and_training_resets_empty_the_store() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let mut inner = fixture(0xDFA5_1604);
    for reset in ["command", "reset_caches_sync", "apply_gradients"] {
        run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(3), true, 12)?;
        ChatBackend::reset_caches(&mut inner, ResetScope::PrefixMiss)?;
        assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{reset}");
        match reset {
            "command" => ChatBackend::reset_caches(&mut inner, ResetScope::Command)?,
            "reset_caches_sync" => inner.reset_caches_sync()?,
            _ => inner.apply_gradients_inner(HashMap::new(), 0.0, &HashMap::new())?,
        }
        assert_eq!(inner.dflash2_states.store.bytes().entries, 0, "{reset}");
        inner.reset_caches_sync()?;
    }
    Ok(())
}

/// Releasing a cache owner after its DFlash2 turn leaves the flat live
/// state alone, so a request from another owner on the same content plus a
/// delta takes the live hit. X1 against the same two turns with no release.
#[test]
fn owner_release_keeps_the_flat_live_state() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let first = first_prompt(4);
    let mut reference = fixture(0xDFA5_1605);
    run_tiny_dflash_turn(&mut reference, &tokenizer, &first, true, 12)?;
    let second = extend(&reference.cached_token_history, &[4, 5, 6]);
    let reference_cached = run_tiny_dflash_turn(&mut reference, &tokenizer, &second, true, 12)?;
    let reference_state = live_state(&reference)?;

    let mut inner = fixture(0xDFA5_1605);
    inner.initialize_paged_adapter()?;
    ChatBackend::set_cache_owner_id(&mut inner, "owner-a", None);
    run_tiny_dflash_turn(&mut inner, &tokenizer, &first, true, 12)?;
    assert_eq!(inner.active_scheduled_seq, None);
    let history = inner.cached_token_history.clone();
    HybridSchedulerBackend::release_owner_resources(&mut inner, Some(1), None)?;
    HybridSchedulerBackend::release_owner_resources(&mut inner, None, None)?;
    ChatBackend::set_cache_owner_id(&mut inner, "owner-b", None);
    let resumed = run_tiny_dflash_turn(&mut inner, &tokenizer, &second, true, 12)?;
    assert_eq!(resumed.cached_tokens, history.len() as u32, "live hit");
    assert_eq!(resumed.cached_tokens, reference_cached.cached_tokens);
    assert!(live_state(&inner)? == reference_state, "X1 after release");
    Ok(())
}

/// A scheduled admission whose page-pool grow is declined parks the idle
/// live DFlash2 state, evicts one state (the least recently used), and grows
/// on the retry; with no DFlash2 state at all the declined grow still
/// rejects.
/// Mutation: skip the park in `reclaim_kv` (`park: false`) → red.
/// Mutation: drop every parked state at once (`store.clear()` in place of
/// `apply_lossy` in `take_state_unit`) → red (0 ≠ 1 entries).
#[test]
fn scheduled_pool_grow_reclaims_one_state_per_decline() -> Result<()> {
    use crate::engine::hybrid_scheduler::tests::reserve_past_a_declined_grow;
    use crate::models::qwen3_5::dflash2_turn::KV_BUDGET_OVERRIDE;
    struct Override;
    impl Drop for Override {
        fn drop(&mut self) {
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(None));
        }
    }
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for case in ["parked", "live", "none"] {
        let mut inner = fixture(0xDFA5_1607);
        inner.initialize_paged_adapter()?;
        let mut live = Vec::new();
        if case != "none" {
            run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(6), true, 12)?;
            live = inner.cached_token_history.clone();
        }
        if case == "parked" {
            run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(7), true, 12)?;
            live = inner.cached_token_history.clone();
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1);
        }
        let override_guard = Override;
        KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
        let (admitted, total_blocks, inner) = reserve_past_a_declined_grow(inner);
        drop(override_guard);
        assert_eq!(admitted, case != "none", "{case}");
        assert_eq!(total_blocks, if case == "none" { 4 } else { 8 }, "{case}");
        let lineages: Vec<Vec<u32>> = inner
            .dflash2_states
            .store
            .entries()
            .map(|entry| entry.tokens().to_vec())
            .collect();
        let want: Vec<Vec<u32>> = if case == "parked" {
            vec![live]
        } else {
            Vec::new()
        };
        assert_eq!(lineages, want, "{case}");
        assert!(inner.dflash2_context.is_none(), "{case}");
    }
    Ok(())
}

/// A continuous-batching activation hands the flat caches to a scheduled
/// sequence. It parks the DFlash2 end state first, whether the scheduler
/// installs the owner's history before activating (a resumed preempted turn)
/// or after (an admission). X0 for the parked buffers; X1 for the DFlash2
/// turn resumed on them against the same two turns run back to back.
#[test]
fn scheduled_activation_parks_the_dflash2_end_state() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let first = first_prompt(5);
    let mut reference = fixture(0xDFA5_1606);
    run_tiny_dflash_turn(&mut reference, &tokenizer, &first, true, 12)?;
    let second = extend(&reference.cached_token_history, &[4, 5, 6]);
    let reference_cached = run_tiny_dflash_turn(&mut reference, &tokenizer, &second, true, 12)?;
    let reference_state = live_state(&reference)?;

    for order in ["activate_first", "install_first"] {
        let mut inner = fixture(0xDFA5_1606);
        inner.initialize_paged_adapter()?;
        run_tiny_dflash_turn(&mut inner, &tokenizer, &first, true, 12)?;
        let before = live_state(&inner)?;
        let owner = vec![1u32, 2, 3];
        if order == "activate_first" {
            HybridSchedulerBackend::activate_scheduled_recurrent(&mut inner, 3)?;
            HybridSchedulerBackend::install_owner_state(&mut inner, 3, &owner);
        } else {
            HybridSchedulerBackend::install_owner_state(&mut inner, 3, &owner);
            HybridSchedulerBackend::activate_scheduled_recurrent(&mut inner, 3)?;
        }
        assert_eq!(inner.active_scheduled_seq, Some(3), "{order}");
        assert_eq!(inner.cached_token_history, owner, "{order}");
        assert_end_hit(&inner, &before.0, order);

        HybridSchedulerBackend::release_scheduled_recurrent_for(&mut inner, 3);
        assert_parked_round_trip(&mut inner, &before, order)?;
        let resumed = run_tiny_dflash_turn(&mut inner, &tokenizer, &second, true, 12)?;
        assert_eq!(
            resumed.cached_tokens, reference_cached.cached_tokens,
            "{order}"
        );
        assert!(
            live_state(&inner)? == reference_state,
            "{order}: X1 resumed turn"
        );
    }
    Ok(())
}

type Probe = Arc<dyn Fn(u32) -> std::result::Result<u32, String> + Send + Sync>;

/// Swaps the model's pool for one of `initial` blocks that grows to `max`,
/// its allocator on the model's clock, with `probe` as the grow probe.
pub(super) fn install_pool(
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

fn passing_probe() -> Probe {
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
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    let block_size = adapter.block_size();
    let mut allocator = adapter.allocator().lock().expect("allocator");
    let blocks: Vec<_> = (0..tokens.len() as u32 / block_size)
        .map(|_| allocator.allocate().expect("a block for the chain"))
        .collect();
    allocator
        .cache_full_blocks(tokens, &blocks, block_size, &[], 0)
        .expect("cache the chain");
    for block in blocks {
        allocator.free(block);
    }
    mlx_paged_attn::chain_hashes(tokens, block_size, &[], 0)
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

/// No compaction and no demotion on this thread until dropped: a store
/// step drops a state.
struct NoKvHeadroom;
impl Drop for NoKvHeadroom {
    fn drop(&mut self) {
        crate::models::qwen3_5::dflash2_turn::KV_BUDGET_OVERRIDE.with(|budget| budget.set(None));
    }
}
fn no_kv_headroom() -> NoKvHeadroom {
    crate::models::qwen3_5::dflash2_turn::KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
    NoKvHeadroom
}

/// One parked DFlash2 state, the live state parked by hand.
fn with_parked_state(seed: u64) -> Result<Qwen35Inner> {
    let tokenizer = tokenizer();
    let mut inner = fixture(seed);
    run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(8), true, 12)?;
    assert!(park_live_dflash2_state(&mut inner));
    assert_eq!(inner.dflash2_states.store.bytes().entries, 1);
    Ok(inner)
}

fn parked_bytes(inner: &Qwen35Inner) -> u64 {
    let bytes = inner.dflash2_states.store.bytes();
    bytes.kv_bytes + bytes.state_bytes
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

/// Below the cap the take goes by each holder's own stamp: an older ladder
/// rung, then the parked state, then a newer history entry.
/// Mutation: offer the store step before the GDN entries in
/// `oldest_state_unit` → red (`[Store, ..]`).
#[test]
fn oldest_state_goes_first_across_holders() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = with_parked_state(0xDFA5_1802)?;
    let _headroom = no_kv_headroom();
    install_pool(&mut inner, 4, 8, probe_after(3))?;
    let bytes_per_block = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .bytes_per_block()
        .map_err(Error::from_reason)?;
    assert!(
        parked_bytes(&inner) >= bytes_per_block,
        "the count must pass"
    );
    let block_size = pool_block_size(&inner);
    hold_blocks(&mut inner, 1, 4);
    push_rung(&mut inner, vec![3; block_size as usize], vec![0xAB], 0);
    inner.gdn_last_history_checkpoint = Some(DenseGdnHistoryCheckpoint {
        owner_id: inner.active_cache_owner_id.clone(),
        image_key: None,
        tokens: vec![5, 6, 7],
        caches: Vec::new(),
        last_used: inner.cache_recency.next(),
    });
    clear_grow_reclaim_log();
    let mut runs = 0;
    paged_step_reclaiming(&mut inner, reserve_step(&mut runs, 1, 5 * block_size))?;
    assert_eq!(runs, 2);
    assert_eq!(
        grow_reclaim_log(),
        vec![
            GrowReclaim::Ladder,
            GrowReclaim::Store,
            GrowReclaim::History
        ]
    );
    Ok(())
}

/// At the cap the older parked state is no unit (it sits on no block), so
/// the newer ladder rung goes alone.
/// Mutation: drop the at-cap store skip in `oldest_state_unit` → red
/// (`[Store, Ladder]`).
#[test]
fn cap_reclaim_takes_a_ladder_entry_by_its_own_stamp() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = with_parked_state(0xDFA5_1803)?;
    install_pool(&mut inner, 4, 4, passing_probe())?;
    let block_size = pool_block_size(&inner);
    let chain = chain_tokens(2, 4, block_size);
    let hashes = cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, chain, hashes, stamp);
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
    assert_eq!(inner.dflash2_states.store.bytes().entries, 1);
    Ok(())
}

/// The allocator, the store and the turn end stamp one clock: a block
/// registered before a turn ranks older than the state it parked, and one
/// registered after ranks newer.
/// Mutation: keep the store on a private clock (drop `share_recency` in
/// `FlatStates::new`) → red.
#[test]
fn store_and_allocator_share_one_clock() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let mut inner = fixture(0xDFA5_1804);
    inner.initialize_paged_adapter()?;
    let block_size = pool_block_size(&inner);
    let first = cache_chain(&inner, &chain_tokens(3, 1, block_size))[0];
    run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(9), true, 12)?;
    assert!(park_live_dflash2_state(&mut inner));
    let second = cache_chain(&inner, &chain_tokens(4, 1, block_size))[0];
    let (parked, _) = inner
        .dflash2_states
        .store
        .next_lossy(u64::MAX, None)
        .expect("the parked state");
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    let allocator = adapter.allocator().lock().expect("allocator");
    let before = allocator.block_stamp(first).expect("first stamp");
    let after = allocator.block_stamp(second).expect("second stamp");
    assert!(
        before < parked && parked < after,
        "{before} {parked} {after}"
    );
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

/// The prompt grow charges no block for the marked chain its prompt hits:
/// with that hit out, the free block covers the need, so no grow is probed
/// and no state goes.
/// Mutation: drop `- marked_hit` in `grow_paged_pool_for_prompt` → red
/// (the grow is probed).
#[test]
fn prompt_grow_counts_no_block_for_its_marked_hit() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let mut inner = with_parked_state(0xDFA5_1809)?;
    let probes = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let calls = Arc::clone(&probes);
    install_pool(
        &mut inner,
        4,
        8,
        Arc::new(move |_| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(4)
        }),
    )?;
    let block_size = pool_block_size(&inner);
    let chain = chain_tokens(14, 3, block_size);
    let hashes = cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    push_rung(&mut inner, chain.clone(), hashes, stamp);
    let mut prompt = chain;
    prompt.extend(chain_tokens(15, 1, block_size));
    let keys = per_block_keys(&prompt, block_size);
    let keep = Keep {
        tokens: &prompt,
        keys: Some((&keys, 0)),
    };
    assert_eq!(inner.marked_hit_blocks(keep), 3);
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .takeable_blocks(),
        1
    );
    ensure_paged_pool_for_prompt(&mut inner, &prompt, &keys, 0);
    assert_eq!(probes.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(inner.dflash2_states.store.bytes().entries, 1);
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
    run_tiny_dflash_turn(&mut inner, &tokenizer, &first_prompt(10), true, 12)?;
    let caches = restorable_caches(&inner);
    inner.dflash2_states.store.clear();
    inner.dflash2_context = None;
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
            inner.gdn_state_bytes(&[]) > 0,
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

/// The dense VLM prepare hits a kept rung's chain whose GDN state cannot
/// restore, so it restarts cold at the cap: the restart is refused until
/// the other rung goes. The candidate prepare fits in the free block, so
/// the restart's reclaim is the only one on the path.
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
    push_rung(&mut inner, other, other_hashes, newer);
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
    assert_eq!(left[0].0 as usize, kept.len());
    Ok(())
}
