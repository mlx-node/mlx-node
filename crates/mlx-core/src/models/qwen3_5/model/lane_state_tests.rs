//! The state tier on a load without a draft: a plain autoregressive turn
//! publishes its plan's points (Splash `publishReachedStateBoundaries`
//! runs for every request), writes a refused one from the lane, copies an
//! ordinary point behind, and resumes from it after a restart.

use std::sync::Arc;

use super::*;
use crate::array::memory::WeightsResident;
use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard};
use crate::disk_tier::{CacheDisk, DiskTierOptions};
use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
use crate::engine::types::{ChatConfig, ChatResult};
use crate::models::qwen3_5::dflash2_decode::tests::{
    metal_is_default_device, tiny_dflash_inner_with_config,
};
use crate::models::qwen3_5::dflash2_state::disk::tests::{
    TestTier, attach_tier, persistent_test_tier, test_tier,
};
use crate::models::qwen3_5::dflash2_state::disk::{
    close_cache, encode_anchor, poll_offload, read_object,
};
use crate::models::qwen3_5::dflash2_state::plan::STATE_WINDOW_OVERRIDE;
use crate::models::qwen3_5::dflash2_state::store::EntryId;
use crate::models::qwen3_5::dflash2_turn::{STATE_CLAIM_SCRIPT, ScriptedClaim};
use crate::models::qwen3_5::model::dflash2_park_tests::{install_pool, passing_probe};
use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;

/// A 32-row state window and 16-row chunks on this thread, so a tiny
/// prompt plans a checkpoint at 64 (two windows) and its replay point.
struct SmallPlan;

impl SmallPlan {
    fn new() -> Self {
        STATE_WINDOW_OVERRIDE.with(|window| window.set(Some(32)));
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        Self
    }
}

impl Drop for SmallPlan {
    fn drop(&mut self) {
        STATE_WINDOW_OVERRIDE.with(|window| window.set(None));
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        STATE_CLAIM_SCRIPT.with(|script| script.borrow_mut().clear());
    }
}

/// The tiny dense model with no draft on a 32-block pool of 16-row blocks.
fn plain(seed: u64, test: &str) -> Option<Qwen35Inner> {
    if !metal_is_default_device() || crate::test_support::paged_prefill_unsupported(test) {
        return None;
    }
    let mut inner = tiny_dflash_inner_with_config(seed, |_| {});
    inner.dflash2 = None;
    install_pool(&mut inner, 32, 32, passing_probe()).expect("pool");
    assert_eq!(inner.paged_adapter.as_ref().expect("pool").block_size(), 16);
    Some(inner)
}

/// The model's states on `tier`, the GDN cell alone, and the adapter's
/// state-backed cold context on it (Splash `openDiskTier`).
fn attach(inner: &mut Qwen35Inner, tier: &TestTier) {
    attach_tier(
        &mut inner.dflash2_states,
        &inner.config,
        inner.paged_adapter.as_ref().expect("pool"),
        tier,
        0,
        0,
    );
    inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
        crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
    );
}

/// Word-level `t0`..`t15`, whose template renders each message as `t1` and
/// the generation prompt as `t2 t3`.
fn tokenizer() -> Arc<Qwen3Tokenizer> {
    let vocab = (0..16)
        .map(|i| format!("\"t{i}\": {i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }},
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "WordLevel",
                "vocab": {{ {vocab} }},
                "unk_token": "t0"
            }}
        }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .unwrap_or_else(|error| panic!("fixture tokenizer: {error}"));
    Arc::new(
        Qwen3Tokenizer::from_tokenizer(
            tokenizer,
            Some(
                "{% for message in messages %}t1 {% endfor %}{% if add_generation_prompt %}t2 \
                 t3{% endif %}"
                    .to_string(),
            ),
        )
        .expect("tokenizer"),
    )
}

fn config() -> ChatConfig {
    ChatConfig {
        temperature: Some(0.0),
        max_new_tokens: Some(2),
        reasoning_effort: Some("none".to_string()),
        ..ChatConfig::default()
    }
}

/// `body` tokens of `seed`, then the template's generation prompt: its
/// replay point is the last whole block of the body (112 for 127 tokens).
fn prompt(seed: u32, body: u32) -> Vec<u32> {
    let mut tokens: Vec<u32> = (0..body).map(|i| (seed + i * 7) % 13 + 1).collect();
    tokens.extend([2, 3]);
    tokens
}

/// One whole autoregressive turn of `tokens` (Splash `admit` → prefill →
/// decode → `release`).
fn ar_turn(inner: &mut Qwen35Inner, tokens: &[u32]) -> Result<ChatResult> {
    whole_turn(inner, tokens, &config(), DecoderPlan::Autoregressive)
}

/// The token ids the last turn on this thread decoded.
fn generated() -> Vec<u32> {
    crate::engine::finalize::LAST_GENERATED.with(|last| last.borrow().clone())
}

/// One whole native-MTP turn of `tokens`, greedy, eight tokens, as a
/// session that leaves `enableMtp` unset runs it on an MTP-head checkpoint.
fn mtp_turn(inner: &mut Qwen35Inner, tokens: &[u32], reuse: bool) -> Result<ChatResult> {
    inner.mtp_weights_loaded = true;
    let config = ChatConfig {
        max_new_tokens: Some(8),
        enable_mtp: Some(true),
        report_performance: Some(true),
        reuse_cache: Some(reuse),
        ..config()
    };
    let result = whole_turn(
        inner,
        tokens,
        &config,
        DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::NativeMtp),
    )?;
    let cycles = result
        .performance
        .as_ref()
        .and_then(|metrics| metrics.mtp_cycles)
        .unwrap_or(0);
    assert!(cycles >= 1, "the turn ran MTP cycles: {cycles}");
    Ok(result)
}

fn whole_turn(
    inner: &mut Qwen35Inner,
    tokens: &[u32],
    config: &ChatConfig,
    decoder: DecoderPlan,
) -> Result<ChatResult> {
    let tokenizer = tokenizer();
    let config = config.clone();
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
        sink: None,
        cancelled: None,
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };
    match ChatBackend::run_paged_turn(inner, &mut args)? {
        TurnOutput::Complete(result) => Ok(*result),
        TurnOutput::Streamed => Err(Error::from_reason("a sync turn streamed")),
    }
}

/// Each stored state: (depth, checkpoint, RAM copy, disk copy).
fn states(inner: &Qwen35Inner) -> Vec<(usize, bool, bool, bool)> {
    let mut states: Vec<_> = inner
        .dflash2_states
        .store
        .entries()
        .map(|entry| {
            (
                entry.depth(),
                entry.checkpoint(),
                entry.ram().is_some(),
                entry.disk().is_some(),
            )
        })
        .collect();
    states.sort_unstable();
    states
}

fn state_at(inner: &Qwen35Inner, depth: usize) -> EntryId {
    inner
        .dflash2_states
        .store
        .entries()
        .find(|entry| entry.depth() == depth)
        .unwrap_or_else(|| panic!("a state at {depth}"))
        .id()
}

/// The GDN cell of `id`'s RAM copy as the tier encodes it.
fn cell_bytes(inner: &Qwen35Inner, id: EntryId) -> Vec<u8> {
    let anchor = inner
        .dflash2_states
        .store
        .get(id)
        .and_then(|entry| entry.ram())
        .expect("a RAM copy");
    encode_anchor(&inner.dflash2_states, &inner.config, anchor).expect("an encoded cell")
}

/// Splash `openDiskTier` for a load without a draft: a pool load whose
/// plan set a state's staging aside opens the tier, its states one span,
/// the GDN cell, its geometry naming no draft.
/// Mutation AT1 (the tier opens only with a draft) → red.
#[test]
fn plain_paged_load_opens_the_state_tier() {
    let Some(mut inner) = plain(0x5A13_0001, "plain_paged_load_opens_the_state_tier") else {
        return;
    };
    let root = std::env::temp_dir().join(format!(
        "mlx-state-tier-plain-tier-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let target = root.join("target");
    std::fs::create_dir_all(&target).expect("target dir");
    std::fs::create_dir_all(root.join("tmp")).expect("temp root");
    std::fs::write(target.join("model.safetensors"), b"target").expect("target shard");
    inner.disk_options = DiskTierOptions {
        cache_disk: CacheDisk::Bytes(1 << 30),
        persistent: false,
        cache_dir: Some(root.join("cache")),
        temp_root: Some(root.join("tmp")),
    };
    let activation =
        crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(&inner).unwrap();
    let cell = crate::models::qwen3_5::gdn_sidecar::geometry(&inner.config, activation)
        .and_then(|geometry| geometry.cell_bytes())
        .expect("a GDN cell");
    let mut plan = crate::engine::memory_plan::MemoryPlan::with_pool_window(0, 1, 0, 512);
    plan.footprint.state_staging_bytes = inner.config.recurrent_state_bytes();
    inner.memory_plan = Some(plan);
    let weights = WeightsResident::for_test(1, 4096);
    let files = CheckpointFiles::Dir(&target);
    let mut guard = CheckpointLoadGuard::before_read(files, true);
    guard.record_read();
    inner.build_disk_tier_context(files, None, &weights, &guard);
    let disk = inner
        .dflash2_states
        .disk
        .as_ref()
        .expect("a plain paged load opens the state tier");
    assert_eq!(disk.spans, vec![cell]);
    assert!(
        disk.geometry.ends_with("draft none\ncontext none\n"),
        "{}",
        disk.geometry
    );
    assert!(disk.tier.is_open() && !disk.tier.persistent);
    disk.tier.close(std::time::Instant::now());
    let _ = std::fs::remove_dir_all(&root);
}

/// Splash `configureDraftStatePlan` for a request with no draft: the
/// checkpoint at two windows (64) and the prompt's replay point, the last
/// whole block before the generation prompt (112), publish as the turn's
/// chunks reach them; the replay point stays ordinary and the checkpoint
/// stays cached at the turn's end (Splash `release` retires nothing).
/// Mutations AS1 (the plain prefill keeps no sink) and AG1 (the whole turn
/// leaves the generation prompt unset, so the point moves to 128) → red.
#[test]
fn plain_ar_prefill_publishes_its_plan() -> Result<()> {
    let Some(mut inner) = plain(0x5A13_0002, "plain_ar_prefill_publishes_its_plan") else {
        return Ok(());
    };
    let _plan = SmallPlan::new();
    let tokens = prompt(5, 127);
    let result = ar_turn(&mut inner, &tokens)?;
    assert_eq!(result.cached_tokens, 0);
    let counters = inner.dflash2_states.counters;
    assert_eq!(counters.checkpoint_publications, 1);
    assert_eq!(counters.replay_state_publications, 1);
    assert_eq!(counters.checkpoint_publication_failures, 0);
    assert_eq!(counters.replay_state_publication_failures, 0);
    assert_eq!(
        states(&inner),
        vec![(64, true, true, false), (112, false, true, false)]
    );
    assert!(inner.dflash2_states.turns.is_empty(), "the turn ended");
    assert!(inner.dflash2_states.replay_points.is_empty());
    Ok(())
}

/// Splash `publishReachedStateBoundaries` → `publishStateToDisk` with
/// `snapshotToDisk`: a checkpoint the ledger refused before its chunk goes
/// from the lane to the tier with no RAM copy, counted as a disk
/// publication; its object holds the cell a granted twin keeps in RAM.
/// Mutation AS2 (the refused point is not written from the lane) → red.
#[test]
fn refused_ar_checkpoint_goes_to_disk_from_the_lane() -> Result<()> {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    let Some(mut inner) = plain(
        0x5A13_0003,
        "refused_ar_checkpoint_goes_to_disk_from_the_lane",
    ) else {
        return Ok(());
    };
    let Some(mut twin) = plain(
        0x5A13_0003,
        "refused_ar_checkpoint_goes_to_disk_from_the_lane",
    ) else {
        return Ok(());
    };
    let tier = test_tier("state-tier-ar-lane", 1 << 30);
    let twin_tier = test_tier("state-tier-ar-lane-twin", 1 << 30);
    attach(&mut inner, &tier);
    attach(&mut twin, &twin_tier);
    let _plan = SmallPlan::new();
    let io = FakeIo::new(
        stats(0),
        host(32 * G, crate::engine::governor::host::Pressure::Normal),
    );
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    STATE_CLAIM_SCRIPT.with(|script| {
        *script.borrow_mut() = [ScriptedClaim::Refuse].into_iter().collect();
    });
    let tokens = prompt(9, 127);
    ar_turn(&mut inner, &tokens)?;
    let counters = inner.dflash2_states.counters;
    assert_eq!(counters.disk_state_publications, 1);
    assert_eq!(counters.checkpoint_publications, 1);
    assert_eq!(counters.checkpoint_publication_failures, 0);
    assert_eq!(
        states(&inner),
        vec![(64, true, false, true), (112, false, true, false)]
    );
    assert!(tier.tier.manager.drain(std::time::Duration::from_secs(10)));
    poll_offload(&mut inner);
    let lane_state = state_at(&inner, 64);
    assert!(
        inner
            .dflash2_states
            .store
            .get(lane_state)
            .and_then(|entry| entry.disk())
            .is_some_and(|copy| copy.landed),
        "the lane's write landed"
    );
    let object = read_object(&inner.dflash2_states, state_at(&inner, 64)).expect("the object");

    ar_turn(&mut twin, &tokens)?;
    assert_eq!(twin.dflash2_states.counters.disk_state_publications, 0);
    assert_eq!(object, cell_bytes(&twin, state_at(&twin, 64)));
    Ok(())
}

/// Splash `WriteBehind::published` → `Cache::persist` and `Cache::adopt` →
/// `pollRestores` → `promotedCopy`: the ordinary replay point waits behind
/// (the checkpoint never does), a clean close writes it and its chain, and
/// after a restart the next load takes it back; a turn of the same prompt
/// resumes at it from disk with the cell the first process stored, and the
/// disk read turns resident under an optional claim.
/// Mutations AW1 (no write-behind without a draft), AD2 (adoption refuses a
/// one-span state) and AP1 (the lane turn skips the promotion) → red.
#[test]
fn ar_restart_restores_the_disk_state() -> Result<()> {
    let Some(mut first) = plain(0x5A13_0005, "ar_restart_restores_the_disk_state") else {
        return Ok(());
    };
    let tier = persistent_test_tier("state-tier-ar-restart", 1 << 30);
    attach(&mut first, &tier);
    let _plan = SmallPlan::new();
    let tokens = prompt(11, 127);
    ar_turn(&mut first, &tokens)?;
    let behind = |inner: &Qwen35Inner| {
        inner
            .dflash2_states
            .disk
            .as_ref()
            .expect("tier")
            .write_behind
            .snapshot()
    };
    assert_eq!(behind(&first).waiting, 1, "the replay point waits behind");
    let stored = cell_bytes(&first, state_at(&first, 112));
    assert!(close_cache(
        &mut first,
        std::time::Instant::now() + std::time::Duration::from_secs(30)
    ));
    assert_eq!(behind(&first).durable, 1);
    drop(first);

    let tier = tier.reopen("state-tier-ar-restart", 1 << 30);
    let Some(mut second) = plain(0x5A13_0005, "ar_restart_restores_the_disk_state") else {
        return Ok(());
    };
    attach(&mut second, &tier);
    assert_eq!(
        crate::models::qwen3_5::dflash2_state::disk::adopt_all(&mut second),
        1
    );
    assert_eq!(states(&second), vec![(112, false, false, true)]);
    let result = ar_turn(&mut second, &tokens)?;
    assert_eq!(result.cached_tokens, 112);
    let disk = second.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(
        disk.counters.promotions + disk.counters.promotions_skipped,
        1
    );
    assert_eq!(disk.counters.invalidations, 0);
    assert_eq!(cell_bytes(&second, state_at(&second, 112)), stored);
    Ok(())
}

/// Splash `Request` per lane: two scheduled sequences whose prefill slices
/// interleave keep their own points, cursors and checkpoints; a slice that
/// ends at a point publishes it from the lane at the command's end (Splash
/// caps a command at the next boundary), and the next slice goes on past it.
/// Mutations TS1 (every lane is sequence 0) and AS3 (a slice's last chunk
/// never captures) → red.
#[test]
fn interleaved_scheduled_lanes_keep_their_own_points() -> Result<()> {
    let Some(mut inner) = plain(
        0x5A13_0007,
        "interleaved_scheduled_lanes_keep_their_own_points",
    ) else {
        return Ok(());
    };
    let _plan = SmallPlan::new();
    let prompts = [(1u32, prompt(3, 138)), (2u32, prompt(4, 138))];
    let mut bases = Vec::new();
    for (seq, tokens) in &prompts {
        inner.activate_scheduled_recurrent(*seq)?;
        let base = PagedBackend::prime_prefix_state(&mut inner, tokens, true, 16, &[], 0)?;
        assert_eq!(base.effective_cached_prefix_len, 0);
        bases.push(base);
        inner.park_active_scheduled_recurrent()?;
    }
    for (start, end) in [(0usize, 64usize), (64, 140)] {
        for ((seq, tokens), base) in prompts.iter().zip(&bases) {
            inner.activate_paged_seq(*seq)?;
            let prefix = HybridSchedulerBackend::build_scheduled_prefix(
                &inner,
                base,
                start,
                end - start,
                tokens.clone(),
                start == 0,
            );
            PagedBackend::paged_prefill(
                &mut inner,
                &tokens[start..end],
                &prefix,
                crate::stream::Stream::default(crate::stream::DeviceType::Gpu),
            )?
            .eval();
            inner.park_active_scheduled_recurrent()?;
        }
    }
    let counters = inner.dflash2_states.counters;
    assert_eq!(counters.checkpoint_publications, 2);
    assert_eq!(counters.replay_state_publications, 2);
    assert_eq!(
        states(&inner),
        vec![
            (64, true, true, false),
            (64, true, true, false),
            (128, false, true, false),
            (128, false, true, false),
        ]
    );
    for (seq, tokens) in &prompts {
        let turn = inner
            .dflash2_states
            .turns
            .get(seq)
            .expect("the lane's turn");
        assert_eq!(turn.cursor, turn.plan.boundaries.len());
        let block = inner
            .dflash2_states
            .replay_points
            .get(seq)
            .copied()
            .expect("the lane's replay point");
        let id = state_at_lineage(&inner, &tokens[..128]);
        assert_eq!(
            inner.dflash2_states.store.get(id).map(|entry| entry.mark()),
            Some(block)
        );
    }
    Ok(())
}

fn state_at_lineage(inner: &Qwen35Inner, tokens: &[u32]) -> EntryId {
    inner
        .dflash2_states
        .store
        .entries()
        .find(|entry| entry.tokens() == tokens)
        .expect("a state on the lineage")
        .id()
}

/// Splash `publishReachedStateBoundaries` runs for every request, a
/// speculative one included: a native-MTP whole turn publishes the plan's
/// checkpoint (64) and replay point (112) as a plain turn does, and one the
/// ledger refused goes from the lane to disk; the token stream is the one a
/// twin turn that keeps no state decodes (publishing reads the lane, never
/// writes it).
/// Mutations MW1 (the MTP prefill keeps no sink), MO1 (the MTP core opens
/// no lane turn), MG1 (the MTP turn leaves the generation prompt unset, so
/// the point moves to 128) and ME1 (the MTP core's finalize keeps the
/// turn's points) → red.
#[test]
fn mtp_turn_publishes_its_plan_and_decodes_as_without_states() -> Result<()> {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    const TEST: &str = "mtp_turn_publishes_its_plan_and_decodes_as_without_states";
    let (Some(mut granted), Some(mut refused), Some(mut stateless)) = (
        plain(0x5A13_0011, TEST),
        plain(0x5A13_0011, TEST),
        plain(0x5A13_0011, TEST),
    ) else {
        return Ok(());
    };
    let tier = test_tier("state-tier-mtp-lane", 1 << 30);
    let granted_tier = test_tier("state-tier-mtp-lane-granted", 1 << 30);
    attach(&mut refused, &tier);
    attach(&mut granted, &granted_tier);
    let _plan = SmallPlan::new();
    let tokens = prompt(13, 127);

    let reference = mtp_turn(&mut stateless, &tokens, false)?;
    let reference_ids = generated();
    assert_eq!(reference_ids.len() as u32, reference.num_tokens);
    let counters = stateless.dflash2_states.counters;
    assert_eq!(
        (
            counters.checkpoint_publications,
            counters.replay_state_publications
        ),
        (0, 0),
        "a turn without reuse plans no point"
    );

    let result = mtp_turn(&mut granted, &tokens, true)?;
    assert_eq!(result.cached_tokens, 0);
    let counters = granted.dflash2_states.counters;
    assert_eq!(counters.checkpoint_publications, 1);
    assert_eq!(counters.replay_state_publications, 1);
    assert_eq!(counters.checkpoint_publication_failures, 0);
    assert_eq!(counters.replay_state_publication_failures, 0);
    assert_eq!(
        states(&granted),
        vec![(64, true, true, false), (112, false, true, false)]
    );
    assert!(granted.dflash2_states.turns.is_empty(), "the turn ended");
    assert!(granted.dflash2_states.replay_points.is_empty());
    assert_eq!(result.raw_text, reference.raw_text, "MTP decodes alike");
    assert_eq!(result.num_tokens, reference.num_tokens);
    assert_eq!(generated(), reference_ids, "MTP decodes the same token ids");

    let io = FakeIo::new(
        stats(0),
        host(32 * G, crate::engine::governor::host::Pressure::Normal),
    );
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
    refused.memory_registration =
        Some(governor.register_model(model(16 * G, 0)).expect("register"));
    STATE_CLAIM_SCRIPT.with(|script| {
        *script.borrow_mut() = [ScriptedClaim::Refuse].into_iter().collect();
    });
    let lane = mtp_turn(&mut refused, &tokens, true)?;
    let counters = refused.dflash2_states.counters;
    assert_eq!(counters.disk_state_publications, 1);
    assert_eq!(counters.checkpoint_publications, 1);
    assert_eq!(
        states(&refused),
        vec![(64, true, false, true), (112, false, true, false)]
    );
    assert!(tier.tier.manager.drain(std::time::Duration::from_secs(10)));
    poll_offload(&mut refused);
    let object = read_object(&refused.dflash2_states, state_at(&refused, 64)).expect("the object");
    assert_eq!(object, cell_bytes(&granted, state_at(&granted, 64)));
    let lane_ids = generated();
    assert_eq!(lane.raw_text, reference.raw_text, "MTP decodes alike");
    assert_eq!(lane.num_tokens, reference.num_tokens);
    assert_eq!(lane_ids, reference_ids, "MTP decodes the same token ids");
    Ok(())
}

/// Splash `WriteBehind`, `Cache::adopt`, `acquireDeepest` and
/// `promotedCopy` for a native-MTP conversation: the MTP turn's replay
/// point waits behind and a clean close writes it; after a restart the
/// next MTP turn of the same prompt arms the disk state's chain with its
/// KV lookup, resumes at 112 with the stored cell, turns the read resident
/// and ends its use of the state with the turn.
/// Mutation MA1 (the MTP core arms no disk restore) → red.
#[test]
fn mtp_restart_restores_the_disk_state() -> Result<()> {
    const TEST: &str = "mtp_restart_restores_the_disk_state";
    let Some(mut first) = plain(0x5A13_0013, TEST) else {
        return Ok(());
    };
    let tier = persistent_test_tier("state-tier-mtp-restart", 1 << 30);
    attach(&mut first, &tier);
    let _plan = SmallPlan::new();
    let tokens = prompt(17, 127);
    mtp_turn(&mut first, &tokens, true)?;
    let behind = |inner: &Qwen35Inner| {
        inner
            .dflash2_states
            .disk
            .as_ref()
            .expect("tier")
            .write_behind
            .snapshot()
    };
    assert_eq!(behind(&first).waiting, 1, "the replay point waits behind");
    let stored = cell_bytes(&first, state_at(&first, 112));
    assert!(close_cache(
        &mut first,
        std::time::Instant::now() + std::time::Duration::from_secs(30)
    ));
    assert_eq!(behind(&first).durable, 1);
    drop(first);

    let tier = tier.reopen("state-tier-mtp-restart", 1 << 30);
    let Some(mut second) = plain(0x5A13_0013, TEST) else {
        return Ok(());
    };
    attach(&mut second, &tier);
    assert_eq!(
        crate::models::qwen3_5::dflash2_state::disk::adopt_all(&mut second),
        1
    );
    assert_eq!(states(&second), vec![(112, false, false, true)]);
    let result = mtp_turn(&mut second, &tokens, true)?;
    assert_eq!(result.cached_tokens, 112);
    let disk = second.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(
        disk.counters.promotions + disk.counters.promotions_skipped,
        1
    );
    assert_eq!(disk.counters.invalidations, 0);
    assert_eq!(cell_bytes(&second, state_at(&second, 112)), stored);
    assert_eq!(second.gdn_in_use_for_test(0), None, "the use ended");
    Ok(())
}

/// Splash `publishReachedStateBoundaries` for a scheduled MTP sequence:
/// its prefill slices reach the lane's points and publish them while the
/// same forward seeds the drafter with the slice's hidden rows.
/// Mutation MS1 (a scheduled MTP prefill keeps no sink) → red.
#[test]
fn scheduled_mtp_prefill_publishes_its_plan() -> Result<()> {
    use crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget;
    let Some(mut inner) = plain(0x5A13_0015, "scheduled_mtp_prefill_publishes_its_plan") else {
        return Ok(());
    };
    inner.mtp_weights_loaded = true;
    let _plan = SmallPlan::new();
    let seq = 3u32;
    let tokens = prompt(19, 138);
    inner.activate_scheduled_recurrent(seq)?;
    let base = PagedBackend::prime_prefix_state(&mut inner, &tokens, true, 16, &[], 0)?;
    assert_eq!(base.effective_cached_prefix_len, 0);
    assert!(inner.begin_scheduled_mtp(seq, 0)?);
    for (start, end) in [(0usize, 64usize), (64, 140)] {
        let prefix = HybridSchedulerBackend::build_scheduled_prefix(
            &inner,
            &base,
            start,
            end - start,
            tokens.clone(),
            start == 0,
        );
        PagedBackend::paged_prefill(
            &mut inner,
            &tokens[start..end],
            &prefix,
            crate::stream::Stream::default(crate::stream::DeviceType::Gpu),
        )?
        .eval();
    }
    let counters = inner.dflash2_states.counters;
    assert_eq!(counters.checkpoint_publications, 1);
    assert_eq!(counters.replay_state_publications, 1);
    assert_eq!(
        states(&inner),
        vec![(64, true, true, false), (128, false, true, false)]
    );
    let owner = &inner.mtp_state().owners[&seq];
    assert!(owner.hidden.is_some(), "the drafter is seeded");
    Ok(())
}

/// Image placeholders at rows 4..36, all of image `hash`: blocks 0-2 carry
/// its keys.
fn image_positions(hash: u64) -> Vec<(u32, u64)> {
    (4..36).map(|row| (row, hash)).collect()
}

/// The per-block keys an image turn of `len` tokens over `hash` looks up
/// and registers with.
fn image_keys(len: usize, hash: u64) -> Vec<Vec<u64>> {
    crate::engine::build_paged_extra_keys(len, 16, &image_positions(hash))
}

/// One image turn of `tokens` over image `hash` as the VLM core runs it:
/// the image-aware prepare, the lane's prefill and the finalize, the
/// merge standing in as the token embeddings on a text M-RoPE grid.
/// Returns the prefix the turn resumed at.
fn image_turn(inner: &mut Qwen35Inner, tokens: &[u32], hash: u64) -> Result<u32> {
    inner.dflash2_states.generation_suffix = Some(Arc::from(&[2u32, 3][..]));
    let keys = image_keys(tokens.len(), hash);
    let len = tokens.len() as u32;
    let resolution =
        inner.prepare_dense_vlm_paged_prefix(tokens, len, 16, &keys, true, false, hash, 0)?;
    let cached = resolution.effective_plan.cached_prefix_len;
    let rows = tokens.len() as i64;
    let merge = crate::engine::vision::VisionMerge {
        inputs_embeds: inner
            .embedding
            .forward(&MxArray::from_uint32(tokens, &[1, rows])?)?,
        position_ids: MxArray::from_int32(
            &(0..3).flat_map(|_| 0..len as i32).collect::<Vec<_>>(),
            &[3, 1, rows],
        )?,
        rope_deltas: 0,
    };
    let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
        inner.config.num_layers as usize,
        |i| inner.config.is_linear_layer(i),
    );
    let (logits, rungs) = inner.vlm_lane_prefill(
        tokens,
        &keys,
        &merge,
        cached,
        resolution.gdn_prefix_already_primed,
        0,
        true,
        &layer_kinds,
    )?;
    logits.eval();
    inner.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(tokens, &keys, 0, rungs);
    inner.finalize_dense_manual_paged_turn(&image_positions(hash), 0, tokens.len())?;
    Ok(cached)
}

/// Splash `publishReachedStateBoundaries` for an image request, and
/// `Cache::adopt` with the blocks' image identity (`KvBlockLabel`): an
/// image turn publishes its plan's points under its blocks' image keys,
/// the replay point is written behind with them, and after a restart a
/// turn of the same placeholder ids over another image restores nothing
/// (the stored state stays), while one over the same image resumes at the
/// replay point with the stored cell.
/// Mutations VK1 (the publication drops the image keys), VK2 (the tier's
/// chain keys drop them), VK3 (the label drops them), VK4 (the image
/// prepare arms no disk restore), VK5 (the image prefill keeps no sink) and
/// VK6 (an image turn registers its blocks as text) → red.
#[test]
fn image_restart_restores_only_the_same_image() -> Result<()> {
    const TEST: &str = "image_restart_restores_only_the_same_image";
    const IMAGE: u64 = 0xA11C_E001;
    const OTHER: u64 = 0xB0B0_E002;
    let Some(mut first) = plain(0x5A13_0021, TEST) else {
        return Ok(());
    };
    let tier = persistent_test_tier("state-tier-image-restart", 1 << 30);
    attach(&mut first, &tier);
    let _plan = SmallPlan::new();
    let tokens = prompt(19, 125);
    assert_eq!(image_turn(&mut first, &tokens, IMAGE)?, 0);
    let counters = first.dflash2_states.counters;
    assert_eq!(
        (
            counters.checkpoint_publications,
            counters.replay_state_publications,
            counters.replay_state_publication_failures,
        ),
        (1, 1, 0)
    );
    assert_eq!(
        states(&first),
        vec![(64, true, true, false), (112, false, true, false)]
    );
    let blocks: Vec<Vec<u64>> = image_keys(tokens.len(), IMAGE)[..3].to_vec();
    for entry in first.dflash2_states.store.entries() {
        assert_eq!(entry.extra_keys(), &blocks[..], "the image keys");
    }
    assert!(first.dflash2_states.turns.is_empty(), "the turn ended");
    let stored = cell_bytes(&first, state_at(&first, 112));
    assert!(close_cache(
        &mut first,
        std::time::Instant::now() + std::time::Duration::from_secs(30)
    ));
    drop(first);

    let tier = tier.reopen("state-tier-image-restart", 1 << 30);
    let Some(mut second) = plain(0x5A13_0021, TEST) else {
        return Ok(());
    };
    attach(&mut second, &tier);
    assert_eq!(
        crate::models::qwen3_5::dflash2_state::disk::adopt_all(&mut second),
        1
    );
    let adopted = state_at(&second, 112);
    assert_eq!(
        second
            .dflash2_states
            .store
            .get(adopted)
            .expect("adopted")
            .extra_keys(),
        &blocks[..]
    );
    assert_eq!(
        image_turn(&mut second, &tokens, OTHER)?,
        0,
        "another image at the same placeholder ids restores nothing"
    );
    let disk = second.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.counters.invalidations, 0);
    assert_eq!(disk.counters.promotions, 0);
    assert!(
        second.dflash2_states.store.get(adopted).is_some(),
        "the image's state stays"
    );
    assert_eq!(image_turn(&mut second, &tokens, IMAGE)?, 112);
    let disk = second.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(
        disk.counters.promotions + disk.counters.promotions_skipped,
        1
    );
    assert_eq!(disk.counters.invalidations, 0);
    assert_eq!(cell_bytes(&second, adopted), stored);
    Ok(())
}

/// Splash `configureDraftStatePlan` for an image request on a draft load:
/// the image turn's states carry the draft window the prefill's taps fill,
/// under the blocks' image keys.
/// Mutation VD1 (an image turn on a draft load keeps the plain sink) → red.
#[test]
fn image_turn_on_a_draft_load_publishes_draft_states() -> Result<()> {
    const TEST: &str = "image_turn_on_a_draft_load_publishes_draft_states";
    if !metal_is_default_device() || crate::test_support::paged_prefill_unsupported(TEST) {
        return Ok(());
    }
    let mut inner = tiny_dflash_inner_with_config(0x5A13_0023, |_| {});
    install_pool(&mut inner, 32, 32, passing_probe()).expect("pool");
    assert!(inner.dflash2.is_some());
    let tokens = prompt(23, 125);
    image_turn(&mut inner, &tokens, 0xA11C_E003)?;
    assert!(inner.dflash2.is_some(), "the draft is back");
    let blocks: Vec<Vec<u64>> = image_keys(tokens.len(), 0xA11C_E003)[..3].to_vec();
    let entries: Vec<_> = inner.dflash2_states.store.entries().collect();
    assert!(!entries.is_empty(), "the image turn published nothing");
    for entry in entries {
        assert!(
            entry.ram().is_some_and(|state| state.window.is_some()),
            "a state at {} carries no draft window",
            entry.depth()
        );
        assert_eq!(entry.extra_keys(), &blocks[..]);
    }
    assert!(inner.dflash2_states.turns.is_empty(), "the turn ended");
    Ok(())
}

/// Each RAM state's depth, window bounds and window rows' bits.
fn state_windows(inner: &Qwen35Inner) -> Result<Vec<(usize, (u32, u32), Vec<u32>)>> {
    let mut windows = Vec::new();
    for entry in inner.dflash2_states.store.entries() {
        let window = entry
            .ram()
            .and_then(|state| state.window.as_ref())
            .unwrap_or_else(|| panic!("a state at {} carries no draft window", entry.depth()));
        let rows = window
            .load_rows(window.begin(), window.end() - window.begin())?
            .astype(crate::array::DType::Float32)?;
        rows.eval();
        let bits = rows
            .to_float32()?
            .iter()
            .map(|value| value.to_bits())
            .collect();
        windows.push((entry.depth(), (window.begin(), window.end()), bits));
    }
    windows.sort_by_key(|window| window.0);
    Ok(windows)
}

/// Splash restores a window with each state it resumes at, so a lane that
/// resumes with none draws no point's window from rows below its resume.
/// An image turn on a draft load whose prompt extends the session's last
/// one resumes at its KV hit (128) on the session's live GDN cell, with no
/// stored state there: it publishes only the points a whole draft window
/// (40) past 128, the replay point (208) and not the checkpoint (160), each
/// with the whole window a cold turn of the same prompt publishes there.
/// Mutation KI1 (`image_lane_prefill` keeps every point) → red.
#[test]
fn image_turn_at_a_kv_hit_without_a_state_publishes_whole_windows() -> Result<()> {
    const TEST: &str = "image_turn_at_a_kv_hit_without_a_state_publishes_whole_windows";
    const IMAGE: u64 = 0xA11C_E005;
    if !metal_is_default_device() || crate::test_support::paged_prefill_unsupported(TEST) {
        return Ok(());
    }
    let draft_load = || {
        let mut inner = tiny_dflash_inner_with_config(0x5A13_0025, |_| {});
        install_pool(&mut inner, 32, 32, passing_probe()).expect("pool");
        inner.dflash2.as_mut().expect("draft").config.sliding_window = 40;
        inner
    };
    let _plan = SmallPlan::new();
    let tokens = prompt(29, 220);
    let mut cold = draft_load();
    assert_eq!(image_turn(&mut cold, &tokens, IMAGE)?, 0);
    let reference = state_windows(&cold)?;
    assert_eq!(
        reference.iter().map(|window| window.0).collect::<Vec<_>>(),
        vec![160, 208],
        "the checkpoint at 160 retired the one at 80 (Splash `retireCheckpoint`)"
    );

    let mut inner = draft_load();
    let head = &tokens[..128];
    assert_eq!(image_turn(&mut inner, head, IMAGE)?, 0);
    assert!(
        inner
            .dflash2_states
            .store
            .entries()
            .all(|entry| entry.depth() < 128),
        "no state at the KV hit"
    );
    inner.cached_token_history = head.to_vec();
    inner.cached_image_key = Some(IMAGE);
    assert_eq!(image_turn(&mut inner, &tokens, IMAGE)?, 128);
    assert!(inner.dflash2.is_some(), "the draft is back");
    let published: Vec<_> = state_windows(&inner)?
        .into_iter()
        .filter(|window| window.0 > 128)
        .collect();
    let expected: Vec<_> = reference
        .into_iter()
        .filter(|window| window.0 == 208)
        .collect();
    assert_eq!(
        published, expected,
        "only whole windows past the resume are published"
    );
    Ok(())
}
