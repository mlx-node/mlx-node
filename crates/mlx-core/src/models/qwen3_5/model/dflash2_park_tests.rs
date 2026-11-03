//! Sites that take the flat lane away from a DFlash2 state park it in the
//! model's state store; resets that end the model's history clear the store.

use std::sync::Arc;

use super::*;
use crate::engine::backend::{ChatBackend, ResetScope, WholeTurnArgs};
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::dflash2_decode::tests::{
    array_fingerprints, metal_is_default_device, run_tiny_dflash_turn,
    tiny_dflash_inner_with_attention_head_dim,
};
use crate::models::qwen3_5::dflash2_state::store::Lineage;
use crate::models::qwen3_5::dflash2_state::trace::HitKind;
use crate::models::qwen3_5::dflash2_turn::install_flat_dflash2_end;

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
