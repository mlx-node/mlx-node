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
        "mlx-s13-plain-tier-{}-{}",
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
    let tier = test_tier("s13-ar-lane", 1 << 30);
    let twin_tier = test_tier("s13-ar-lane-twin", 1 << 30);
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
    let tier = persistent_test_tier("s13-ar-restart", 1 << 30);
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

    let tier = tier.reopen("s13-ar-restart", 1 << 30);
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
    let tier = test_tier("s13-mtp-lane", 1 << 30);
    let granted_tier = test_tier("s13-mtp-lane-granted", 1 << 30);
    attach(&mut refused, &tier);
    attach(&mut granted, &granted_tier);
    let _plan = SmallPlan::new();
    let tokens = prompt(13, 127);

    let reference = mtp_turn(&mut stateless, &tokens, false)?;
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
    assert_eq!(lane.raw_text, reference.raw_text, "MTP decodes alike");
    assert_eq!(lane.num_tokens, reference.num_tokens);
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
    let tier = persistent_test_tier("s13-mtp-restart", 1 << 30);
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

    let tier = tier.reopen("s13-mtp-restart", 1 << 30);
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
