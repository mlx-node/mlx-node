use super::*;
use crate::engine::plan::{TurnPath, TurnPlan, TurnRequest};
use crate::models::gemma4::output_parser::{StreamSegment, parse_gemma4_output};

#[test]
fn grouped_cold_finalize_recovers_a_missing_scheduler_snapshot_from_live_state() {
    let request_tokens = (0..160).collect::<Vec<u32>>();
    let mut reads = Vec::new();
    let checkpoint = resolve_grouped_sliding_cold_checkpoint(
        None,
        &[64, 256, 1024],
        96,
        &request_tokens,
        |boundary| {
            reads.push(boundary);
            Ok(Some(Vec::new()))
        },
    )
    .unwrap()
    .unwrap();

    assert_eq!(reads, vec![64]);
    assert_eq!(checkpoint.boundary, 64);
    assert_eq!(checkpoint.tokens, request_tokens[..64]);
    assert!(checkpoint.layer_kv.is_empty());
}

#[test]
fn grouped_cold_finalize_does_not_fabricate_a_rotated_out_anchor() {
    let request_tokens = (0..640).collect::<Vec<u32>>();
    let checkpoint = resolve_grouped_sliding_cold_checkpoint(
        None,
        &[64, 256, 1024],
        512,
        &request_tokens,
        |_boundary| Ok(None),
    )
    .unwrap();

    assert!(checkpoint.is_none());
}

#[test]
fn sliding_sidecar_chain_isolated_by_cache_salt() {
    let fingerprint = mlx_paged_attn::ColdCacheFingerprint::from_components([
        b"gemma4-sidecar-salt-test".as_slice(),
    ]);
    let tokens: Vec<u32> = (1..=8).collect();
    let extra_keys = vec![Vec::new(), Vec::new()];
    let key = |salt| {
        gemma4_sliding_cold_sidecar_chain_key(fingerprint, &tokens, &extra_keys, 4, 8, salt)
            .expect("two-block sidecar chain")
    };

    assert_ne!(key(0), key(11));
    assert_ne!(key(11), key(22));
    assert_eq!(key(11), key(11));
}

#[test]
fn test_gemma4_session_media_payload_identity() {
    let images = vec![vec![1, 2, 3]];
    let audio = vec![vec![4, 5, 6]];
    let image_key = Some(engine::compute_image_cache_key(&images));
    let audio_key = Some(engine::compute_image_cache_key(&audio));

    assert!(gemma4_session_media_matches_payloads(
        true,
        image_key,
        None,
        &images,
        &[]
    ));
    assert!(!gemma4_session_media_matches_payloads(
        false,
        image_key,
        None,
        &images,
        &[]
    ));
    assert!(!gemma4_session_media_matches_payloads(
        true, image_key, audio_key, &images, &audio
    ));
    assert!(!gemma4_session_media_matches_payloads(
        true,
        image_key,
        None,
        &[vec![1, 2, 4]],
        &[]
    ));
    assert!(!gemma4_session_media_matches_payloads(
        true,
        None,
        audio_key,
        &[],
        &[vec![4, 5, 7]]
    ));
    assert!(!gemma4_session_media_matches_payloads(
        true,
        None,
        None,
        &images,
        &[]
    ));
}

#[test]
fn prompt_holds_media_placeholders_detects_image_audio_and_text() {
    let image_token_id = 258880u32;
    let audio_token_id = 258881u32;

    let image_prompt = [1u32, 2, image_token_id, 3];
    assert!(prompt_holds_media_placeholders(
        &image_prompt,
        image_token_id,
        audio_token_id
    ));

    let audio_prompt = [4u32, audio_token_id, 5];
    assert!(prompt_holds_media_placeholders(
        &audio_prompt,
        image_token_id,
        audio_token_id
    ));

    let text_prompt = [6u32, 7, 8, 9];
    assert!(!prompt_holds_media_placeholders(
        &text_prompt,
        image_token_id,
        audio_token_id
    ));
}

#[test]
fn gemma4_media_plan_separates_availability_from_backend_validation() {
    let text_only_flat = gemma4_media_plan(false, false, false);
    assert_eq!(text_only_flat.available, MediaCapabilities::NONE);
    assert_eq!(text_only_flat.backend_validated, MediaCapabilities::IMAGES);

    let image_flat = gemma4_media_plan(true, false, false);
    assert_eq!(image_flat.available, MediaCapabilities::NONE);
    assert_eq!(image_flat.backend_validated, MediaCapabilities::IMAGES);

    let audio_flat = gemma4_media_plan(false, true, false);
    assert_eq!(audio_flat.available, MediaCapabilities::NONE);
    assert_eq!(
        audio_flat.backend_validated,
        MediaCapabilities::IMAGES_AND_AUDIO
    );

    let media_paged = gemma4_media_plan(true, true, true);
    assert_eq!(media_paged.available, MediaCapabilities::IMAGES_AND_AUDIO);
    assert_eq!(media_paged.backend_validated, MediaCapabilities::NONE);

    let missing_image_components_paged = gemma4_media_plan(false, true, true);
    assert_eq!(
        missing_image_components_paged.available,
        MediaCapabilities {
            images: false,
            audio: true,
        }
    );
    assert_eq!(
        missing_image_components_paged.backend_validated,
        MediaCapabilities::IMAGES
    );
}

#[test]
fn gemma4_image_capability_requires_one_complete_paged_path() {
    assert!(gemma4_image_path_loaded(true, true, true, false, true));
    assert!(gemma4_image_path_loaded(true, true, false, true, true));

    assert!(!gemma4_image_path_loaded(false, true, true, false, true));
    assert!(!gemma4_image_path_loaded(true, false, true, false, true));
    assert!(!gemma4_image_path_loaded(true, true, false, false, true));
    assert!(!gemma4_image_path_loaded(true, true, true, false, false));
}

#[test]
fn gemma4_image_lineage_requires_declared_media_context() {
    let history = [1, 2, 3, 4];
    let extended = [1, 2, 3, 4, 5];
    let image_positions = [(1, 0xAAAA)];

    assert!(gemma4_carries_image_lineage(
        MediaCapabilities::IMAGES,
        Some(0xAAAA),
        &image_positions,
        &history,
        &extended,
    ));
    assert!(!gemma4_carries_image_lineage(
        MediaCapabilities::NONE,
        Some(0xAAAA),
        &image_positions,
        &history,
        &extended,
    ));
    assert!(!gemma4_carries_image_lineage(
        MediaCapabilities::IMAGES,
        Some(0xAAAA),
        &image_positions,
        &history,
        &[1, 2, 9],
    ));
}

#[test]
fn gemma4_causal_leading_text_hit_replays_only_before_image() {
    let before_image = gemma4_vlm_prefix_policy(16, Some(32), None);
    assert!(before_image.unified_boundary_safe);
    assert!(!before_image.require_exact_checkpoint);
    assert!(before_image.may_replay_leading_text);

    let at_image_boundary = gemma4_vlm_prefix_policy(32, Some(32), None);
    assert!(!at_image_boundary.require_exact_checkpoint);
    assert!(at_image_boundary.may_replay_leading_text);

    let crosses_image = gemma4_vlm_prefix_policy(48, Some(32), None);
    assert!(crosses_image.require_exact_checkpoint);
    assert!(!crosses_image.may_replay_leading_text);

    let unified_inside_image = gemma4_vlm_prefix_policy(48, Some(32), Some(80));
    assert!(!unified_inside_image.unified_boundary_safe);
    assert!(unified_inside_image.require_exact_checkpoint);
    assert!(!unified_inside_image.may_replay_leading_text);

    let unified_after_image = gemma4_vlm_prefix_policy(80, Some(32), Some(80));
    assert!(unified_after_image.unified_boundary_safe);
    assert!(unified_after_image.require_exact_checkpoint);
    assert!(!unified_after_image.may_replay_leading_text);
}

#[test]
fn gemma4_sliding_cold_capture_context_is_fail_closed_for_media() {
    let image_positions = [(47, 0xAAAA), (32, 0xBBBB), (79, 0xAAAA)];

    assert_eq!(
        Gemma4SlidingColdCaptureContext::text(128, &[]).minimum_safe_boundary(),
        Some(0),
        "the existing text-only capture has no media floor"
    );
    assert_eq!(
        Gemma4SlidingColdCaptureContext::text(128, &image_positions).minimum_safe_boundary(),
        None,
        "a generic text turn carrying image lineage must remain unsupported"
    );
    assert_eq!(
        Gemma4SlidingColdCaptureContext::pure_image(128, &[]).minimum_safe_boundary(),
        None,
        "a pure-image label without image positions must not capture"
    );
    assert_eq!(
        Gemma4SlidingColdCaptureContext::pure_image(128, &image_positions).minimum_safe_boundary(),
        Some(80),
        "the floor must sit after the complete image run even if positions arrive unsorted"
    );
    assert_eq!(
        Gemma4SlidingColdCaptureContext::pure_image(u32::MAX, &[(u32::MAX, 0xAAAA)],)
            .minimum_safe_boundary(),
        None,
        "an unrepresentable exclusive image endpoint must fail closed"
    );
}

#[test]
fn gemma4_unified_first_chunk_never_splits_inside_image_overlay() {
    assert_eq!(
        gemma4_vlm_prefill_chunk_end(0, 128, 32, true, 0, 48, Some(80)),
        128,
        "an inside-image prompt checkpoint must be ignored"
    );
    assert_eq!(
        gemma4_vlm_prefill_chunk_end(0, 128, 32, true, 0, 96, Some(80)),
        96,
        "a checkpoint after the complete image span is safe"
    );
    assert_eq!(
        gemma4_vlm_prefill_chunk_end(0, 128, 32, false, 16, 48, Some(80)),
        16,
        "causal E2B may still split at a leading-text checkpoint"
    );
}

#[test]
fn gemma4_large_sliding_snapshots_are_memory_bounded() {
    let mut config = paged_tiny_config(None);
    config.num_hidden_layers = 40;
    config.layer_types = vec!["sliding_attention".to_string(); 40];
    config.num_kv_shared_layers = None;
    config.sliding_window = 1024;
    config.num_key_value_heads = 8;
    config.head_dim = 256;

    assert_eq!(
        gemma4_sliding_checkpoint_estimated_bytes(&config),
        40 * 1024 * 8 * 256 * 2 * 4
    );
    assert_eq!(
        gemma4_sliding_prefix_checkpoint_limit_for_override(&config, 16, None),
        2,
        "the default byte budget must not retain 128 huge unified snapshots"
    );
    assert_eq!(
        gemma4_sliding_retention_caps_for_override(&config, 16, false, None),
        Gemma4SlidingRetentionCaps::pre_ladder(
            2,
            Gemma4SlidingCheckpointBytes::for_config(&config)
        ),
        "a persistence-OFF turn must keep the pre-ladder cap verbatim"
    );
    assert_eq!(
        gemma4_sliding_retention_caps_for_override(&config, 16, true, None),
        Gemma4SlidingRetentionCaps::ladder(
            6,
            Gemma4SlidingAnchorRungs::from_slice(&[64, 256, 1024, 4096]),
            Gemma4SlidingCheckpointBytes::for_config(&config)
        ),
        "a persist turn widens by exactly the anchor rung count"
    );
}

/// A hybrid geometry with every sixth layer global, by the four axes that
/// move the checkpoint byte arithmetic.
fn sliding_config(
    num_hidden_layers: i32,
    sliding_window: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    num_kv_shared_layers: Option<i32>,
) -> super::Gemma4Config {
    let mut config = paged_tiny_config(None);
    config.num_hidden_layers = num_hidden_layers;
    config.layer_types = (0..num_hidden_layers)
        .map(|index| {
            if (index + 1) % 6 == 0 {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();
    config.num_kv_shared_layers = num_kv_shared_layers;
    config.sliding_window = sliding_window;
    config.num_key_value_heads = num_key_value_heads;
    config.head_dim = head_dim;
    config
}

/// The geometry that produced this bug on real weights:
/// `Gemma-4-12B-IT-nvidia-mxfp-mlx` — 48 decoder layers, every sixth
/// global, so 40 physical sliding layers; window 1024; 8 kv heads;
/// head_dim 256; no KV sharing.
fn twelve_b_sliding_config() -> super::Gemma4Config {
    sliding_config(48, 1024, 8, 256, None)
}

/// Geometries the byte cap must hold on besides the 12B.
///
/// These are NOT claims about `Gemma-4-26B-A4B` or `Gemma-4-E2B`
/// specifically: this repo carries no config for either and neither was
/// available locally, so encoding one under that name would be a guess
/// dressed as a fixture. What they do encode are the AXES a second geometry
/// moves — KV sharing (which turns trailing sliding layers into aliases and
/// so shrinks a checkpoint), a narrower window with fewer/smaller heads
/// (which makes checkpoints cheap enough that the COUNT cap already fits the
/// budget), and an all-global stack (no sliding state at all, where the byte
/// cap must be an inert no-op rather than a divide-by-zero). Pinning the
/// invariant across all four is the point; pinning it on one shape is how
/// the count cap came to be treated as a byte cap in the first place.
fn kv_shared_sliding_config() -> super::Gemma4Config {
    sliding_config(48, 1024, 8, 256, Some(4))
}

fn narrow_window_sliding_config() -> super::Gemma4Config {
    sliding_config(30, 512, 4, 128, None)
}

fn all_global_config() -> super::Gemma4Config {
    let mut config = sliding_config(30, 512, 4, 128, None);
    config.layer_types = vec!["full_attention".to_string(); 30];
    config
}

/// A draft, not a checkpoint: the anchor flag is not a thing a publish site
/// (or a test standing in for one) can set. `into_checkpoint` derives it
/// from the caps, and a test that set it by hand would be testing its own
/// bookkeeping instead of the seam the prefill actually goes through.
fn sliding_checkpoint_at(
    prefix_len: u32,
    block_size: u32,
    tokens: &[u32],
) -> Gemma4SlidingPrefixCheckpointDraft {
    Gemma4SlidingPrefixCheckpointDraft {
        prefix_len,
        block_size,
        final_block_hash: u64::from(prefix_len),
        protected_image_prompt_boundary: false,
        tokens: tokens[..prefix_len as usize].to_vec(),
        snapshots: Vec::new(),
    }
}

/// The same draft, flagged as a VLM prompt boundary — what
/// `remember_gemma4_sliding_materialized_prompt_boundary_checkpoint_with_keys`
/// stores on an image turn. Never an eviction candidate for steps 1 or 2 of
/// the ladder victim rule, which is what makes the deep fallback reachable.
fn image_prompt_checkpoint_at(
    prefix_len: u32,
    block_size: u32,
    tokens: &[u32],
) -> Gemma4SlidingPrefixCheckpointDraft {
    Gemma4SlidingPrefixCheckpointDraft {
        protected_image_prompt_boundary: true,
        ..sliding_checkpoint_at(prefix_len, block_size, tokens)
    }
}

/// Replay one prefill's checkpoint pushes through the real publish +
/// retention seams, into a store that may already hold an earlier turn's
/// entries. Returns the boundaries the chunk loop snapshotted at; what is
/// left in `retained` afterwards is the state
/// `capture_gemma4_sliding_cold_sidecar` runs against.
///
/// The chunk walk mirrors `run_paged_prefill_chunk`'s pass-1 loop: the body
/// is forwarded in `chunk_tokens`-sized pieces, each piece publishes
/// `gemma4_sliding_chunk_checkpoint_boundaries` minus the prompt boundary,
/// and the prompt boundary is stored last by its own path.
///
/// The rung list is handed over on BOTH arms, unlike the call site (which
/// does not bother computing it when the ladder is off). Refusing to
/// publish is then the `want_ladder` parameter's own job, which is what a
/// future refactor hoisting the rung computation out of its `if` must not
/// be able to break silently.
///
/// `start_offset` is the pass-1 loop's `chunk_first_position`, i.e. the
/// `cached_prefix_len` a WARM turn resumes from. It is not cosmetic: the
/// rung filter is strict (`rung > start_offset`), so a warm turn republishes
/// none of the rungs below where it resumed, and the store it inherits is
/// the only thing that still holds them.
fn replay_prefill_into(
    retained: &mut VecDeque<Gemma4SlidingPrefixCheckpoint>,
    config: &super::Gemma4Config,
    block_size: u32,
    start_offset: u32,
    prompt_boundary: u32,
    chunk_tokens: u32,
    want_ladder: bool,
) -> Vec<u32> {
    let caps = gemma4_sliding_retention_caps_for_override(config, block_size, want_ladder, None);
    let interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
    let tokens: Vec<u32> = (0..prompt_boundary).collect();

    let mut published: Vec<u32> = Vec::new();
    let mut start = start_offset;
    while start < prompt_boundary {
        let end = (start + chunk_tokens).min(prompt_boundary);
        let mut boundaries = gemma4_sliding_chunk_checkpoint_boundaries(start, end, interval, caps);
        boundaries.retain(|boundary| *boundary != prompt_boundary);
        assert!(
            boundaries.windows(2).all(|pair| pair[0] < pair[1]),
            "prepare_sliding_checkpoint_capture rejects a non-ascending set: {boundaries:?}"
        );
        for boundary in boundaries {
            published.push(boundary);
            upsert_gemma4_sliding_prefix_checkpoint(
                retained,
                sliding_checkpoint_at(boundary, block_size, &tokens),
                caps,
                false,
            );
        }
        start = end;
    }
    published.push(prompt_boundary);
    upsert_gemma4_sliding_prefix_checkpoint(
        retained,
        sliding_checkpoint_at(prompt_boundary, block_size, &tokens),
        caps,
        false,
    );
    published
}

fn retained_boundaries(retained: &VecDeque<Gemma4SlidingPrefixCheckpoint>) -> Vec<u32> {
    retained
        .iter()
        .map(|checkpoint| checkpoint.prefix_len)
        .collect()
}

/// One COLD prefill into an empty store, the shape every offset-0 test wants.
fn replay_prefill_checkpoints(
    config: &super::Gemma4Config,
    block_size: u32,
    prompt_boundary: u32,
    chunk_tokens: u32,
    want_ladder: bool,
) -> (Vec<u32>, Vec<u32>) {
    let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
    let published = replay_prefill_into(
        &mut retained,
        config,
        block_size,
        0,
        prompt_boundary,
        chunk_tokens,
        want_ladder,
    );
    (published, retained_boundaries(&retained))
}

/// Deepest retained boundary a cold capture could anchor on, given the
/// persisted K/V chain only reaches `chain_reach_tokens`. This is the
/// selection `find_gemma4_sliding_capture_checkpoints` performs.
fn deepest_reachable(retained: &[u32], chain_reach_tokens: u32) -> Option<u32> {
    retained
        .iter()
        .copied()
        .filter(|boundary| *boundary <= chain_reach_tokens)
        .max()
}

/// Every boundary a later restore of a `prompt_len`-token prompt would
/// probe, ascending.
///
/// A restated COPY of the READ path, never a call into the capture-side
/// helpers it exists to pin — the same discipline
/// `cold_tier_parity_harness::expected_checkpoint_ladder` follows. A test
/// that derived its expectation from
/// `gemma4_cold_restore_reachable_boundary` would move with that function
/// and could never fail, which is exactly how the one-block gap survived
/// three green suites.
///
/// ```text
///   Gemma4Inner::prepare_gemma4_paged_turn
///       max_cache_hit_tokens = total_budget - 1
///   PagedKVCacheAdapter::find_cached_prefix_per_block_inner
///       lookup_len    = min(max_cache_hit_tokens, prompt_tokens.len())
///       lookup_tokens = &prompt_tokens[..lookup_len]
///   ColdTierWalk::restore_extend
///       full_blocks = lookup_tokens.len() / block_size
///   ColdTierWalk::deepest_backed_boundary
///       for count in (floor + 1..=keys.len()).rev()
///           boundary = count * block_size
/// ```
fn restore_probeable_boundaries(prompt_len: u32, block_size: u32) -> Vec<u32> {
    if block_size == 0 {
        return Vec::new();
    }
    let lookup_len = prompt_len.saturating_sub(1);
    let full_blocks = lookup_len / block_size;
    (1..=full_blocks).map(|count| count * block_size).collect()
}

/// The capture may never anchor where the restore cannot look.
///
/// With the persisted chain unbounded, the ceiling this pins is the ONLY
/// thing standing between the capture and a boundary no key on the read
/// side ever spells. `request_tokens` is deliberately swept past the prompt:
/// the capture runs at finalize, so it sees the completion too, and the
/// defect was precisely that it measured its ceiling against that longer
/// sequence.
#[test]
fn the_capture_ceiling_is_exactly_the_deepest_boundary_a_restore_can_probe() {
    for block_size in [1u32, 8, 16, 32, 64] {
        for prompt_len in 0..=400u32 {
            let probeable = restore_probeable_boundaries(prompt_len, block_size);
            for generated in [0usize, 1, 15, 512] {
                let request_tokens_len = prompt_len as usize + generated;
                let ceiling_blocks = gemma4_sliding_cold_capture_ceiling_blocks(
                    u32::MAX,
                    request_tokens_len,
                    prompt_len,
                    block_size,
                );
                let ceiling_tokens = ceiling_blocks as u32 * block_size;
                match probeable.last() {
                    None => assert_eq!(
                        ceiling_tokens, 0,
                        "block_size={block_size} prompt_len={prompt_len} \
                         generated={generated}: a restore of this prompt can probe NO \
                         boundary, so a capture that names {ceiling_tokens} writes an \
                         object nothing can ask for"
                    ),
                    Some(&deepest) => assert_eq!(
                        ceiling_tokens, deepest,
                        "block_size={block_size} prompt_len={prompt_len} \
                         generated={generated}: the restore probes {probeable:?}; the \
                         capture ceiling must be its deepest member, not \
                         {ceiling_tokens}"
                    ),
                }
            }
        }
    }
}

/// The one-block gap, with the numbers it was measured at.
///
/// A 4-token A/B on Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx, everything else held
/// constant: the 6572-token prompt restored 6560 of 6572 tokens, the
/// 6576-token one restored ZERO. The only difference is that 6576 is a
/// multiple of 16, which puts the prompt-boundary checkpoint one block above
/// `max_cache_hit_tokens`.
#[test]
fn a_block_aligned_prompt_is_the_only_case_the_prompt_boundary_outruns_the_restore() {
    const BS: u32 = 16;
    for (prompt_len, prompt_boundary, reachable) in
        [(6572u32, 6560u32, 6560u32), (6576, 6576, 6560)]
    {
        // What the prompt-boundary publisher aims at
        // (`prompt_checkpoint_boundary_len` in `run_paged_prefill_chunk`).
        assert_eq!(prompt_len / BS * BS, prompt_boundary);
        assert_eq!(
            gemma4_cold_restore_reachable_boundary(prompt_len, BS),
            reachable
        );
        let probeable = restore_probeable_boundaries(prompt_len, BS);
        assert!(
            probeable.contains(&reachable),
            "prompt_len={prompt_len}: the reachable boundary must be one the restore \
             actually enumerates"
        );
        assert_eq!(
            probeable.contains(&prompt_boundary),
            prompt_boundary == reachable,
            "prompt_len={prompt_len}: the prompt boundary {prompt_boundary} is probeable \
             if and only if it IS the reachable one; when it is not, a sidecar anchored \
             there is dead on arrival and self-locking"
        );

        // And the ceiling the capture actually runs with, for the turn as
        // it happened: a 6576-token prompt with a 40-token completion and a
        // chain that covered every block of it.
        let ceiling_blocks = gemma4_sliding_cold_capture_ceiling_blocks(
            u32::MAX,
            prompt_len as usize + 40,
            prompt_len,
            BS,
        );
        assert_eq!(
            ceiling_blocks as u32 * BS,
            reachable,
            "prompt_len={prompt_len}: the capture must stop at {reachable}, the deepest \
             boundary a restore of this prompt enumerates. It measured \
             {} instead, which is what wrote 209.7 MB a restore could never name.",
            ceiling_blocks as u32 * BS
        );
    }
}

/// What the reachability clamp COSTS a growing conversation, and the bound
/// on that cost.
///
/// The clamp is scoped to the whole capture, not just to the aligned prompt
/// boundary it was written for, so it also drops every candidate the DECODE
/// published — `maybe_remember_gemma4_sliding_decode_boundary_checkpoint`
/// publishes over `request_tokens` = prompt + generated, so those
/// candidates are real, and turn N+1 of a growing conversation, whose
/// prompt contains them, really could name them.
///
/// It is kept broad anyway, and this test is the ledger for that choice.
/// One sidecar is written per turn, so the clamp is a PRIORITY rule: it
/// spends the turn's single write on the deepest boundary a restore of THIS
/// prompt can name — the replay a cold tier exists for, and the case that
/// measured zero reuse — instead of on a deeper boundary that pays off only
/// if the conversation continues with exactly these tokens. The give-up is
/// bounded by one turn: turn N+1's own ceiling covers everything turn N
/// discarded, so the deeper boundaries are lost only to a process that dies
/// between the two finalizes.
#[test]
fn the_capture_ceiling_gives_up_this_turns_generated_region_and_the_next_turn_covers_it() {
    const BS: u32 = 16;
    // Turn N: an aligned 6576-token prompt and a 2048-token completion,
    // long enough that the decode cadence published inside it.
    let prompt_n = 6576u32;
    let request_n = prompt_n as usize + 2048;
    let unclamped_n = (request_n as u32 / BS) * BS;
    let ceiling_n =
        gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, request_n, prompt_n, BS) as u32 * BS;
    assert_eq!(
        ceiling_n, 6560,
        "the capture stops at the deepest boundary a restore of THIS prompt enumerates"
    );
    assert_eq!(
        unclamped_n, 8624,
        "…while the chain and the request alone would have allowed 8624"
    );
    assert_eq!(
        unclamped_n - ceiling_n,
        2064,
        "so the clamp gives up 2064 tokens' worth of generated-region candidates"
    );

    // Turn N+1 of the same conversation: its prompt is turn N's whole
    // request plus new user text, so its own ceiling sits at or past
    // everything turn N gave up.
    for new_user_tokens in [1u32, 17, 512] {
        let prompt_next = request_n as u32 + new_user_tokens;
        let ceiling_next = gemma4_sliding_cold_capture_ceiling_blocks(
            u32::MAX,
            prompt_next as usize + 8,
            prompt_next,
            BS,
        ) as u32
            * BS;
        assert!(
            ceiling_next >= unclamped_n,
            "new_user_tokens={new_user_tokens}: turn N+1 must be able to name every \
             boundary turn N discarded ({ceiling_next} < {unclamped_n}), or the clamp \
             loses them for good instead of deferring them by one turn"
        );
    }
}

/// Which prompts need the extra cold-restore tail checkpoint at all, and
/// where it must sit.
///
/// Exactly the block-aligned ones, exactly one block below the prompt
/// boundary. Everywhere else the two coincide and the prefill publishes
/// nothing extra — which is what keeps the added snapshot from being a cost
/// every turn pays.
#[test]
fn the_cold_tail_checkpoint_is_needed_exactly_when_the_prompt_is_block_aligned() {
    for block_size in [1u32, 8, 16, 32] {
        for prompt_len in 1..=300u32 {
            let prompt_boundary = prompt_len / block_size * block_size;
            let tail = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
            if prompt_len.is_multiple_of(block_size) {
                assert_eq!(
                    tail + block_size,
                    prompt_boundary,
                    "block_size={block_size} prompt_len={prompt_len}: an aligned prompt \
                     needs a tail one block below its own boundary"
                );
            } else {
                assert_eq!(
                    tail, prompt_boundary,
                    "block_size={block_size} prompt_len={prompt_len}: a ragged prompt's \
                     boundary is already reachable, so nothing extra may be published"
                );
            }
        }
    }
}

/// THE persistence-OFF transparency claim for this change, as a test.
///
/// Chunk length is the GEMM's `M`, and the retained checkpoint set decides
/// which one a later warm turn resumes from, so both are observable in the
/// emitted tokens. A turn with no `SlidingWindow` sidecar policy must
/// therefore snapshot exactly what it snapshotted before the cold tier
/// existed: nothing extra, at any prompt length, at any block size.
#[test]
fn a_persistence_off_turn_publishes_no_cold_restore_tail_at_all() {
    let config = twelve_b_sliding_config();
    for block_size in [1u32, 8, 16, 32] {
        let off = gemma4_sliding_retention_caps_for_override(&config, block_size, false, None);
        let on = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
        assert!(!off.wants_ladder() && on.wants_ladder());
        for prompt_len in 0..=300u32 {
            assert_eq!(
                gemma4_cold_restore_tail_publish(prompt_len, block_size, off),
                None,
                "block_size={block_size} prompt_len={prompt_len}: a persistence-OFF turn \
                 that snapshots one extra boundary changes the retained set, and with it \
                 the depth a later warm turn resumes from"
            );
            let reachable = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
            assert_eq!(
                gemma4_cold_restore_tail_publish(prompt_len, block_size, on),
                (prompt_len.is_multiple_of(block_size) && reachable > 0).then_some(reachable),
                "block_size={block_size} prompt_len={prompt_len}"
            );
        }
    }
}

/// Where the tail lands inside the prefill's chunk walk, and — the half that
/// keeps it numerically inert — that it never displaces a boundary the
/// chunk was already going to snapshot.
#[test]
fn the_cold_restore_tail_is_captured_beside_the_chunks_own_boundaries_never_instead() {
    // A 6576-token aligned prompt: the body runs to position 6575, so the
    // tail at 6560 falls inside the final chunk.
    let tail = gemma4_cold_restore_tail_publish(
        6576,
        16,
        gemma4_sliding_retention_caps_for_override(&twelve_b_sliding_config(), 16, true, None),
    );
    assert_eq!(tail, Some(6560));

    // Earlier chunks pass it by.
    assert_eq!(gemma4_chunk_cold_restore_tail(tail, 0, 2048, &[1024]), None);
    assert_eq!(
        gemma4_chunk_cold_restore_tail(tail, 2048, 4096, &[3072]),
        None
    );
    // The chunk that crosses it takes it.
    assert_eq!(
        gemma4_chunk_cold_restore_tail(tail, 4096, 6575, &[5120, 6144]),
        Some(6560)
    );
    // `(start, end]`, so a chunk that merely STARTS there does not re-take
    // it — `prepare_sliding_checkpoint_capture` needs strictly increasing
    // offsets and a duplicate would be rejected.
    assert_eq!(
        gemma4_chunk_cold_restore_tail(tail, 6560, 6575, &[]),
        None,
        "a boundary at or below where the chunk began was already passed"
    );
    // And when the cadence or a rung already lands on it, the tail adds
    // nothing: the retained set must stay byte-for-byte the persist-off one.
    assert_eq!(
        gemma4_chunk_cold_restore_tail(tail, 4096, 6575, &[5120, 6144, 6560]),
        None,
        "the tail must never be routed to the singleton INSTEAD of the store when the \
         chunk was already publishing that boundary — that would silently remove an entry \
         a persistence-OFF turn retains"
    );
}

/// A RAGGED prompt must publish no tail at all, and the chunk walk cannot
/// be the thing that enforces it.
///
/// On 15 prompts out of 16 the reachable boundary and the prompt boundary
/// are the same number, and the prompt boundary is already snapshotted by
/// `maybe_remember_gemma4_sliding_prompt_boundary_checkpoint` — which the
/// chunk plan is split at, so that path always fires when the tail would
/// have. Publishing the tail as well takes a SECOND full sliding-window
/// snapshot of one offset, and `find_gemma4_sliding_capture_checkpoints`
/// then dedups the pair back down to one candidate. Cost with no reader.
///
/// The chunk walk cannot catch it: `run_paged_prefill_chunk` strips the
/// prompt boundary out of `checkpoint_boundaries` one line before handing
/// that same list in as `already_published`, so
/// `gemma4_chunk_cold_restore_tail`'s containment test is blind to exactly
/// the coinciding case. The publish gate is where it has to be decided.
#[test]
fn a_ragged_prompt_publishes_no_tail_beside_the_prompt_boundary_it_coincides_with() {
    const BS: u32 = 16;
    let caps =
        gemma4_sliding_retention_caps_for_override(&twelve_b_sliding_config(), BS, true, None);

    // The measured pair: 6572 ragged, 6576 aligned, same block size.
    assert_eq!(
        6572 / BS * BS,
        gemma4_cold_restore_reachable_boundary(6572, BS)
    );
    assert_eq!(
        gemma4_cold_restore_tail_publish(6572, BS, caps),
        None,
        "a ragged prompt's own boundary IS the reachable one and is already \
         snapshotted; a tail here is a duplicate window nothing reads"
    );
    assert_eq!(
        gemma4_cold_restore_tail_publish(6576, BS, caps),
        Some(6560),
        "the aligned prompt is the one case the tail exists for and must keep it"
    );

    // What the prefill would do with it. The chunk that ends on the ragged
    // prompt boundary is the same chunk the prompt-boundary path fires
    // after, and the stripped `already_published` cannot say so.
    let ragged_tail = gemma4_cold_restore_tail_publish(6572, BS, caps);
    assert_eq!(
        gemma4_chunk_cold_restore_tail(ragged_tail, 4096, 6560, &[5120, 6144]),
        None,
        "the chunk that ends on the prompt boundary must publish nothing extra"
    );

    // And it is not one fixture: exactly the aligned prompts publish.
    for block_size in [1u32, 8, 16, 32] {
        let caps = gemma4_sliding_retention_caps_for_override(
            &twelve_b_sliding_config(),
            block_size,
            true,
            None,
        );
        for prompt_len in 1..=300u32 {
            let published =
                gemma4_cold_restore_tail_publish(prompt_len, block_size, caps).is_some();
            let reachable = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
            assert_eq!(
                published,
                prompt_len.is_multiple_of(block_size) && reachable > 0,
                "block_size={block_size} prompt_len={prompt_len}: the tail exists only \
                 where the prompt boundary outruns the restore, i.e. only on an aligned \
                 prompt"
            );
        }
    }
}

/// The descent must step PAST what is already on disk, or a poisoned root
/// never heals.
///
/// The scenario is the one users already have: a pre-fix run anchored a
/// sidecar at the aligned prompt boundary 6576, which no restore can name.
/// Everything shallower is still missing. A walk that stops at the first
/// already-persisted candidate writes nothing, this turn and every turn
/// after it, because the key it recomputes is the same key.
#[test]
fn the_capture_descends_past_boundaries_already_on_disk() {
    // Deepest first, exactly as `find_gemma4_sliding_capture_checkpoints`
    // hands them over.
    let candidates = [6576u32, 6560, 4096, 1024];
    let on_disk = [6576u32];

    assert_eq!(
        gemma4_select_cold_capture_candidate(candidates, |boundary| {
            if on_disk.contains(boundary) {
                Gemma4ColdCaptureProbe::Persisted
            } else {
                Gemma4ColdCaptureProbe::Missing(*boundary)
            }
        }),
        Gemma4ColdCaptureSelection::Capture {
            candidate: 6560,
            key: 6560,
            skipped_persisted: 1,
        },
        "the deepest candidate is the useless one already on disk; the capture must go on \
         to 6560, the boundary a restore of this prompt actually enumerates, and account \
         for the one it passed over"
    );

    // Steady state: everything reachable is already written, so the turn
    // does nothing and says so.
    assert_eq!(
        gemma4_select_cold_capture_candidate(candidates, |_| {
            Gemma4ColdCaptureProbe::<u32>::Persisted
        }),
        Gemma4ColdCaptureSelection::AllPersisted {
            skipped_persisted: candidates.len(),
        },
        "a fully populated ladder must write nothing, not re-enqueue its shallowest rung"
    );

    // A boundary whose chain cannot be derived is not a skip: nothing was
    // ever written there to skip, and counting it as one would make a
    // healthy short turn wear the signature of a saturated ladder.
    assert_eq!(
        gemma4_select_cold_capture_candidate(candidates, |boundary| match *boundary {
            6576 => Gemma4ColdCaptureProbe::Underivable,
            6560 => Gemma4ColdCaptureProbe::Persisted,
            other => Gemma4ColdCaptureProbe::Missing(other),
        }),
        Gemma4ColdCaptureSelection::Capture {
            candidate: 4096,
            key: 4096,
            skipped_persisted: 1,
        }
    );
}

/// A descent that derived NO chain at all must not reach the counters
/// wearing the saturated ladder's signature.
///
/// `Underivable` and `Persisted` are opposite states of the tier — one
/// means nothing was ever written at that boundary, the other means
/// something was — and the capture records a different counter for each.
/// When the walk returned `(None, 0)` for both, the all-`Underivable` turn
/// bumped `already_persisted`, so a root holding nothing reported itself
/// full.
#[test]
fn an_all_underivable_descent_is_not_an_already_persisted_one() {
    let candidates = [6576u32, 6560, 4096, 1024];

    assert_eq!(
        gemma4_select_cold_capture_candidate(candidates, |_| {
            Gemma4ColdCaptureProbe::<u32>::Underivable
        }),
        Gemma4ColdCaptureSelection::NoChainDerived,
        "not one chain derived, so the tier holds nothing here — reporting this as an \
         already-persisted descent makes an empty root read as a saturated one"
    );

    // Mixed, and still not a persistence claim: one derivable boundary that
    // IS on disk is what separates the two outcomes.
    assert_eq!(
        gemma4_select_cold_capture_candidate(candidates, |boundary| match *boundary {
            1024 => Gemma4ColdCaptureProbe::Persisted,
            _ => Gemma4ColdCaptureProbe::<u32>::Underivable,
        }),
        Gemma4ColdCaptureSelection::AllPersisted {
            skipped_persisted: 1,
        }
    );

    // An empty candidate list derives nothing either.
    assert_eq!(
        gemma4_select_cold_capture_candidate([0u32; 0], |_| {
            Gemma4ColdCaptureProbe::<u32>::Underivable
        }),
        Gemma4ColdCaptureSelection::NoChainDerived
    );
}

/// The reachability clamp is an EXTRA bound, not a replacement: the
/// persisted chain and the request still cap the capture, and a turn whose
/// prompt length was never recorded captures nothing rather than guessing.
#[test]
fn the_capture_ceiling_still_honours_the_chain_the_request_and_a_missing_prompt() {
    assert_eq!(
        gemma4_sliding_cold_capture_ceiling_blocks(3, 6616, 6576, 16),
        3,
        "a chain that reached 3 blocks bounds the capture at 3 blocks"
    );
    assert_eq!(
        gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, 32, 6576, 16),
        2,
        "a request holding 2 whole blocks cannot anchor deeper than 2"
    );
    assert_eq!(
        gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, 6576, 0, 16),
        0,
        "no recorded prompt length must fail CLOSED, never fall back to the request"
    );
    assert_eq!(
        gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, 6576, 6576, 0),
        0,
        "block_size 0 folds to a no-op instead of dividing by zero"
    );
}

#[test]
fn gemma4_sliding_anchor_rungs_are_powers_of_four_from_the_block_size() {
    let config = twelve_b_sliding_config();
    assert_eq!(
        gemma4_sliding_cold_anchor_rungs(&config, 16, 2),
        vec![64, 256, 1024, 4096],
        "the grid is block_size * 4^k, pinned to zero so the same rung is \
         reusable by every later turn sharing the prefix"
    );

    // Why the fourth rung fits at all: a rung's payload is min(b, window)
    // rows, so the two sub-window rungs are nearly free. Charging every
    // entry a full window - what `gemma4_sliding_checkpoint_estimated_bytes`
    // does, and all any pre-ladder caller needed - does not fit.
    let full_window = gemma4_sliding_checkpoint_estimated_bytes(&config);
    let reserve = full_window * 2;
    let actual: u64 = [64u32, 256, 1024, 4096]
        .iter()
        .map(|rung| gemma4_sliding_checkpoint_estimated_bytes_at(&config, *rung))
        .sum();
    assert!(
        actual + reserve <= GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
        "boundary-scaled: {} + {} > {}",
        actual,
        reserve,
        GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES
    );
    assert!(
        full_window * 4 + reserve > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
        "flat full-window sizing would have to refuse a rung"
    );
}

#[test]
fn gemma4_sliding_checkpoint_bytes_scale_with_min_boundary_window() {
    let config = twelve_b_sliding_config();
    let full_window = gemma4_sliding_checkpoint_estimated_bytes(&config);
    assert_eq!(
        gemma4_sliding_checkpoint_estimated_bytes_at(&config, 64),
        full_window / 16,
        "a 64-token rung carries 64 of the window's 1024 rows"
    );
    assert_eq!(
        gemma4_sliding_checkpoint_estimated_bytes_at(&config, 1024),
        full_window
    );
    assert_eq!(
        gemma4_sliding_checkpoint_estimated_bytes_at(&config, 4096),
        full_window,
        "past the window a payload stops growing"
    );
}

/// The headline gate for the gemma4 cold-tier ladder.
///
/// Reproduced twice on real weights before the fix
/// (`Gemma-4-12B-IT-nvidia-mxfp-mlx`, 8140-token prompt, `mlx agent`):
///
/// ```text
///   W1 cold     chain reach  576 tok (36 blk)   0 sliding_window sidecars
///   W2 restart  chain reach 1136 tok (71 blk)   0 sliding_window sidecars
///   trace: sliding_cold_sidecar_capture_skipped
///          reason=no_representable_checkpoint_at_or_below_chain_reach
/// ```
///
/// The store finished at `{7168, 8128}` both times: the cadence fires every
/// window, `limit` is 2 on this geometry, and the pre-ladder victim is the
/// oldest entry — so the rung at 1024 was born and then evicted, and
/// nothing at or below the chain's reach was left.
#[test]
fn gemma4_sliding_ladder_retains_a_rung_the_lagging_chain_can_reach() {
    let config = twelve_b_sliding_config();
    let (published, retained) = replay_prefill_checkpoints(&config, 16, 8128, 2048, true);
    assert_eq!(
        published,
        vec![64, 256, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8128],
        "the cadence, plus the anchor rungs, in one ascending set per chunk"
    );
    assert_eq!(
        retained,
        vec![64, 256, 1024, 4096, 7168, 8128],
        "the anchors must survive the cadence ratcheting deeper entries in behind them"
    );
    assert_eq!(
        deepest_reachable(&retained, 576),
        Some(256),
        "turn 1: the chain reached 36 blocks, so only a sub-window rung can anchor"
    );
    assert_eq!(
        deepest_reachable(&retained, 1136),
        Some(1024),
        "turn 2: the chain reached 71 blocks and the rung at 1024 must still be there"
    );
    assert_eq!(
        deepest_reachable(&retained, 4200),
        Some(4096),
        "later turns must keep deepening rather than sticking at one window"
    );
}

/// Lesson (a) from qwen3.5's GDN ladder, which shipped broken twice: a
/// request with no cold tier must retain exactly what it retained before
/// the ladder existed. Which checkpoint a later warm turn lands on decides
/// whether `prepare_gemma4_sliding_prefix` installs a snapshot or replays
/// the whole cached prefix, and those emit different tokens.
#[test]
fn gemma4_persistence_off_retains_exactly_the_pre_ladder_set() {
    let config = twelve_b_sliding_config();
    let caps = gemma4_sliding_retention_caps_for_override(&config, 16, false, None);
    assert_eq!(
        caps,
        Gemma4SlidingRetentionCaps::pre_ladder(
            gemma4_sliding_prefix_checkpoint_limit_for_override(&config, 16, None),
            Gemma4SlidingCheckpointBytes::for_config(&config)
        )
    );
    let (published, retained) = replay_prefill_checkpoints(&config, 16, 8128, 2048, false);
    assert_eq!(
        published,
        vec![1024, 2048, 3072, 4096, 5120, 6144, 7168, 8128],
        "no cold tier means the bare cadence: no rung may be snapshotted"
    );
    assert_eq!(
        retained,
        vec![7168, 8128],
        "and it is trimmed oldest-first to the pre-ladder cap, as it always was"
    );
}

#[test]
fn gemma4_sliding_published_boundaries_are_unchanged_when_the_ladder_is_off() {
    let config = twelve_b_sliding_config();
    // Same rungs on both arms, so refusing to publish them is the POLICY's
    // job. A `PreLadder` turn does not even compute a grid at the call
    // site; handing it one here is what makes this discriminating.
    let anchors =
        Gemma4SlidingAnchorRungs::from_slice(&gemma4_sliding_cold_anchor_rungs(&config, 16, 2));
    assert!(anchors.len > 0);
    let off = Gemma4SlidingRetentionCaps {
        anchors,
        ..gemma4_sliding_retention_caps_for_override(&config, 16, false, None)
    };
    let on = gemma4_sliding_retention_caps_for_override(&config, 16, true, None);
    for (start, end) in [(0u32, 2048u32), (2048, 4096), (4096, 6144), (6144, 8128)] {
        assert_eq!(
            gemma4_sliding_chunk_checkpoint_boundaries(start, end, 1024, off),
            gemma4_sliding_checkpoint_boundaries_crossed(start, end, 1024),
            "chunk ({start}, {end}] must publish the bare cadence with the ladder off"
        );
    }
    assert_eq!(
        gemma4_sliding_chunk_checkpoint_boundaries(0, 2048, 1024, on),
        vec![64, 256, 1024, 2048],
        "with the ladder on, a rung that coincides with the cadence is published once"
    );
}

/// Every cursor in `first..=last` at which decode publishes a checkpoint.
/// `first` is the token count the prefill left behind, since decode only
/// ever walks forward from there.
fn decode_published_boundaries(
    config: &super::Gemma4Config,
    block_size: u32,
    caps: Gemma4SlidingRetentionCaps,
    first: u32,
    last: u32,
) -> Vec<u32> {
    let interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
    (first..=last)
        .filter(|&cursor| gemma4_sliding_decode_publishes_checkpoint(cursor, interval, caps))
        .collect()
}

/// Decode is the ONLY publisher for the shape `mlx agent` actually sends —
/// a short prompt and a long generation — and before this it fired on the
/// cadence alone.
///
/// The cadence is `max(window, block).div_ceil(block) * block` = 1024 here,
/// and `window / block_size = 64 = 4^3`, so the rung ladder and the cadence
/// COLLIDE at every rung with `k >= 3` and miss each other everywhere below:
///
/// ```text
///   rungs    64   256   1024   4096
///   cadence              1024   4096  (…every 1024)
///   union    64   256   1024   4096
/// ```
///
/// A 200-token prompt publishes {64} at prefill and then generates. Without
/// the union, 256 is published by nothing: the cadence skips it, and the
/// next turn's prefill starts past it and its rung filter is strict
/// (`rung > start_offset`). So the chain — which advances ~34 blocks
/// (~544 tokens) per turn — has nothing at or below its reach to anchor on,
/// which is exactly the inert cold tier the ladder exists to fix.
#[test]
fn gemma4_sliding_decode_publishes_the_rungs_the_cadence_skips() {
    let config = twelve_b_sliding_config();
    let caps = gemma4_sliding_retention_caps_for_override(&config, 16, true, None);
    assert_eq!(
        gemma4_sliding_decode_checkpoint_interval(&config, 16),
        1024,
        "the cadence is a whole window, which is ABOVE two of the four rungs"
    );
    assert_eq!(caps.anchors.as_slice(), &[64, 256, 1024, 4096]);
    assert_eq!(
        decode_published_boundaries(&config, 16, caps, 1, 1200),
        vec![64, 256, 1024],
        "the cadence UNION the rungs — the two sub-window rungs are the whole point"
    );
    assert!(
        !gemma4_sliding_decode_publishes_checkpoint(0, 1024, caps),
        "an empty request publishes nothing"
    );
}

/// Defect A: a checkpoint that genuinely sits on a rung but was born with
/// `cold_anchor_rung` clear is the ladder's PREFERRED eviction victim, so
/// the rung the decode path just published is the FIRST thing thrown away.
///
/// This drives the decode publisher (short prompt, long generation) through
/// the same store seam production uses, and asserts the rungs outlive the
/// deeper cadence entries that ratchet in behind them.
#[test]
fn gemma4_sliding_decode_rungs_survive_the_cadence_ratcheting_past_them() {
    let config = twelve_b_sliding_config();
    let block_size = 16;
    let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
    let tokens: Vec<u32> = (0..6000).collect();
    let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();

    // Turn 1: a 200-token prompt. Only the shallowest rung is crossed.
    let published = replay_prefill_into(&mut retained, &config, block_size, 0, 200, 2048, true);
    assert_eq!(published, vec![64, 200]);

    // Then generate to 6000. Every cursor the decode predicate accepts is
    // stored, exactly as `maybe_remember_gemma4_sliding_decode_boundary_checkpoint`
    // does.
    let decode_boundaries = decode_published_boundaries(&config, block_size, caps, 201, 6000);
    assert_eq!(
        decode_boundaries,
        vec![256, 1024, 2048, 3072, 4096, 5120],
        "256 is published by decode or by nothing at all"
    );
    for boundary in &decode_boundaries {
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut retained,
            sliding_checkpoint_at(*boundary, block_size, &tokens),
            caps,
            false,
        );
    }

    let survivors = retained_boundaries(&retained);
    assert_eq!(
        survivors,
        vec![64, 256, 1024, 3072, 4096, 5120],
        "the rungs are deferred; the plain cadence entries are what gets evicted"
    );
    let flagged: Vec<u32> = retained
        .iter()
        .filter(|checkpoint| checkpoint.cold_anchor_rung)
        .map(|checkpoint| checkpoint.prefix_len)
        .collect();
    assert_eq!(
        flagged,
        vec![64, 256, 1024, 4096],
        "a rung published by DECODE must carry the flag too, not only one \
         published by the prefill capture path"
    );
    assert_eq!(
        deepest_reachable(&survivors, 544),
        Some(256),
        "the chain advances ~34 blocks a turn; only a sub-window rung is in reach"
    );
}

/// Defect B's other half, and the axis `replay_prefill_checkpoints` could
/// not see while it hard-coded `start = 0`: a WARM turn resumes at
/// `cached_prefix_len`, and `gemma4_sliding_chunk_checkpoint_boundaries`
/// filters `rung > start_offset`, so it republishes none of the rungs below
/// where it resumed. The inherited store is the only thing that still holds
/// them, and `Ladder` retention is the only reason it still does.
#[test]
fn gemma4_sliding_warm_turn_keeps_the_rungs_it_cannot_republish() {
    let config = twelve_b_sliding_config();
    let block_size = 16;
    let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();

    let turn1 = replay_prefill_into(&mut retained, &config, block_size, 0, 512, 2048, true);
    assert_eq!(turn1, vec![64, 256, 512]);

    let turn2 = replay_prefill_into(&mut retained, &config, block_size, 512, 8128, 2048, true);
    assert_eq!(
        turn2,
        vec![1024, 2048, 3072, 4096, 5120, 6144, 7168, 8128],
        "resuming at 512 republishes no rung below 512"
    );

    let survivors = retained_boundaries(&retained);
    assert_eq!(
        survivors,
        vec![64, 256, 1024, 4096, 7168, 8128],
        "the shallow rungs turn 2 could not republish must survive it"
    );
    assert_eq!(
        deepest_reachable(&survivors, 544),
        Some(256),
        "otherwise a warm turn silently loses everything the chain can reach"
    );
}

/// Defect C: the ladder's `limit` is a COUNT derived from a byte budget on
/// the assumption that the extra slots hold cheap sub-window rungs —
/// `gemma4_sliding_cold_anchor_rungs` prices a 64-token rung at 41.9 MB
/// rather than 671.1 MB, which is the only reason a fourth rung fit. Nothing
/// forces the retained set to BE that mix. Once the cursor is past one
/// window every retained entry costs a full window:
///
/// ```text
///   6 x 671.1 MB = 4026 MB   vs   budget 3072 MB    (+31%)
/// ```
///
/// On unified memory that gigabyte is not taken from a spare tier; it comes
/// out of the weights and the paged pool (see `docs/architecture.md`), and
/// an oversized pool separately costs ~10x on long-context decode.
#[test]
fn gemma4_sliding_ladder_bounds_the_retained_set_in_bytes_not_entries() {
    let block_size = 16u32;
    let mut geometries_that_overran_the_count_cap = 0usize;
    for (label, config) in [
        ("12B", twelve_b_sliding_config()),
        ("kv-shared", kv_shared_sliding_config()),
        ("narrow-window", narrow_window_sliding_config()),
        ("all-global", all_global_config()),
    ] {
        let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
        let window = config.sliding_window.max(0) as u32;
        let full_window = gemma4_sliding_checkpoint_estimated_bytes(&config);
        let tokens: Vec<u32> = (0..u32::from(u16::MAX)).collect();

        // Every entry sits PAST the window, so each costs a full window, and
        // none lands on a rung — the count cap alone would keep all of them.
        let deep: Vec<u32> = (1..=caps.limit as u32)
            .map(|index| window + block_size * index)
            .collect();
        assert!(
            deep.iter()
                .all(|boundary| !caps.anchors.contains(*boundary)),
            "{label}: the scenario must be plain deep entries, not rungs"
        );

        let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
        for boundary in &deep {
            upsert_gemma4_sliding_prefix_checkpoint(
                &mut retained,
                sliding_checkpoint_at(*boundary, block_size, &tokens),
                caps,
                false,
            );
        }

        let retained_bytes = caps.bytes.total(retained.iter());
        assert!(
            retained_bytes <= GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
            "{label}: retained {} entries for {} bytes, over the declared {} ceiling",
            retained.len(),
            retained_bytes,
            GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES
        );
        assert!(
            !retained.is_empty(),
            "{label}: the byte cap must bound the set, not empty it"
        );

        let count_only_bytes = caps.limit as u64 * full_window;
        if count_only_bytes > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES {
            geometries_that_overran_the_count_cap += 1;
            assert!(
                retained.len() < caps.limit,
                "{label}: {} full-window entries are {} bytes, so the byte cap had to evict",
                caps.limit,
                count_only_bytes
            );
        }
    }
    assert!(
        geometries_that_overran_the_count_cap > 0,
        "no geometry actually exercised the overrun; the assertion above would be vacuous"
    );
}

/// The byte budget must not be paid for out of the one rung the chain can
/// reach.
///
/// `gemma4_sliding_ladder_victim` skips anchors at step 1 and skips
/// ancestor-anchors at step 2, but its pre-ladder FLOOR is
/// `position(|c| !protected_image_prompt_boundary)` — the SHALLOWEST entry,
/// anchor or not. Reaching the floor is not exotic: the two
/// `GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT` slots are never eligible for
/// steps 1 or 2, so a store of `{image, image, rung, rung, rung, deep}` has
/// nothing for either step to take. That store is what a VLM turn followed
/// by a fresh text turn leaves behind: `save_paged_history` clears
/// `cached_paged_image_token_positions` on a fresh text turn, so the media
/// refusal in `capture_gemma4_sliding_cold_sidecar` lifts, while the
/// protected entries stay in the store.
///
/// ```text
///   store            img@2048  img@3072   256    1024   4096   deep@5120
///   bytes (MB)          671.1     671.1  167.8   671.1  671.1      671.1
///   total 3523.2 MB  >  3072 MB ceiling  ->  the byte loop must evict
///
///   shallowest-first   evicts 256 then 1024   ->  chain@544 reaches NOTHING
///   deepest-anchor     evicts 4096            ->  chain@544 reaches 256
/// ```
///
/// Evicting the deepest anchor is the cheap answer as well as the right
/// one: one eviction clears the overrun where the shallow rungs take two.
#[test]
fn gemma4_sliding_ladder_byte_budget_never_evicts_the_shallowest_reachable_rung() {
    let config = twelve_b_sliding_config();
    let block_size = 16u32;
    let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
    assert_eq!(caps.limit, 6);
    assert_eq!(caps.anchors.as_slice(), &[64, 256, 1024, 4096]);
    let tokens: Vec<u32> = (0..8192).collect();

    let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
    for draft in [
        image_prompt_checkpoint_at(2048, block_size, &tokens),
        image_prompt_checkpoint_at(3072, block_size, &tokens),
        sliding_checkpoint_at(256, block_size, &tokens),
        sliding_checkpoint_at(1024, block_size, &tokens),
        sliding_checkpoint_at(4096, block_size, &tokens),
        sliding_checkpoint_at(5120, block_size, &tokens),
    ] {
        upsert_gemma4_sliding_prefix_checkpoint(&mut retained, draft, caps, false);
    }

    // The scenario has to be the byte loop and nothing else: the count loop
    // never fires (six entries against a limit of six), and the six of them
    // genuinely overrun the ceiling.
    assert!(
        caps.bytes.at(2048) * 5 + caps.bytes.at(256) > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
        "the pushed set must exceed the ceiling, or this proves nothing"
    );

    let survivors: Vec<(u32, bool, bool)> = retained
        .iter()
        .map(|checkpoint| {
            (
                checkpoint.prefix_len,
                checkpoint.cold_anchor_rung,
                checkpoint.protected_image_prompt_boundary,
            )
        })
        .collect();
    assert_eq!(
        survivors,
        vec![
            (2048, false, true),
            (3072, false, true),
            (256, true, false),
            (1024, true, false),
            (5120, false, false),
        ],
        "the byte loop must take the DEEPEST anchor (4096). Taking the shallowest \
         re-creates the very failure the anchor flag exists to prevent, and taking \
         index 0 throws away a protected image boundary the count loop is required \
         to keep"
    );
    assert!(
        caps.bytes.total(retained.iter()) <= GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
        "and it must actually get under the ceiling"
    );

    let boundaries = retained_boundaries(&retained);
    assert_eq!(
        deepest_reachable(&boundaries, 544),
        Some(256),
        "the persisted chain advances ~34 blocks (~544 tokens) a turn; the shallow \
         rung is the only thing it can anchor on"
    );
}

/// Persistence-OFF, the byte axis. `PreLadder` carries the SAME per-entry
/// cost model as `Ladder` (see `Gemma4SlidingRetentionCaps::bytes`), so the
/// only thing keeping the byte cap off a persistence-OFF turn is `policy`.
/// An override of 8 on the 12B geometry is 5120 MB — well over the ladder's
/// 3072 MB ceiling — and it must still retain all 8, because a smaller
/// retained set moves which checkpoint a later warm turn resumes from and
/// that changes emitted tokens.
#[test]
fn gemma4_persistence_off_is_never_trimmed_by_the_ladder_byte_budget() {
    let config = twelve_b_sliding_config();
    let block_size = 16u32;
    let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, false, Some(8));
    assert_eq!(caps.limit, 8);
    assert!(!caps.wants_ladder());
    assert!(
        caps.bytes.full_window_bytes > 0,
        "the cost model must be populated on this arm, or the guard below \
         would hold for a second, silent reason"
    );

    let tokens: Vec<u32> = (0..u32::from(u16::MAX)).collect();
    let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
    let deep: Vec<u32> = (1..=8u32).map(|index| 1024 + block_size * index).collect();
    for boundary in &deep {
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut retained,
            sliding_checkpoint_at(*boundary, block_size, &tokens),
            caps,
            false,
        );
    }
    assert!(
        caps.bytes.total(retained.iter()) > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
        "the scenario must actually exceed the ladder budget, or this proves nothing"
    );
    assert_eq!(
        retained_boundaries(&retained),
        deep,
        "a persistence-OFF turn retains exactly what it retained before the ladder existed"
    );
}

/// Persistence-OFF, the decode-publish axis. Same construction as
/// `gemma4_sliding_published_boundaries_are_unchanged_when_the_ladder_is_off`:
/// the OFF caps are handed the real rung grid, so refusing to publish is the
/// policy's job and not an accident of an empty list.
#[test]
fn gemma4_persistence_off_decode_publishes_only_the_cadence() {
    let config = twelve_b_sliding_config();
    let anchors =
        Gemma4SlidingAnchorRungs::from_slice(&gemma4_sliding_cold_anchor_rungs(&config, 16, 2));
    assert_eq!(anchors.as_slice(), &[64, 256, 1024, 4096]);
    let off = Gemma4SlidingRetentionCaps {
        anchors,
        ..gemma4_sliding_retention_caps_for_override(&config, 16, false, None)
    };
    assert_eq!(
        decode_published_boundaries(&config, 16, off, 1, 1200),
        vec![1024],
        "with no cold tier the decode cadence is untouched: no rung may fire"
    );
}

/// A real [`ColdTierContext`] carrying `sidecar_policy`, plus the temp root
/// it owns so the caller can remove it.
///
/// Deliberately the real type with a real manager rather than a stand-in:
/// the thing under test is production's own derivation, and a stand-in for
/// `ColdTierContext` would be a second implementation of the fact these
/// tests exist to pin. Opening the manager touches a directory and nothing
/// else — no block is ever written.
fn cold_tier_context_with(
    label: &str,
    sidecar_policy: Option<mlx_paged_attn::ColdSidecarPolicy>,
) -> (ColdTierContext, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "mlx-gemma4-sliding-ladder-{}-{label}",
        std::process::id()
    ));
    let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.clone())
        .expect("temp-dir cold cache must open");
    (
        ColdTierContext {
            manager: Arc::new(manager),
            fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                b"gemma4-sliding-ladder-test".as_slice(),
            ]),
            sidecar_policy,
        },
        root,
    )
}

/// A cold tier belonging to a DIFFERENT hybrid family (qwen3.5's GDN
/// recurrent state). Present so "wants a ladder" cannot be satisfied by
/// "some sidecar policy exists": gemma4's rungs are readable only by
/// gemma4's own sliding capture.
fn gdn_sidecar_policy() -> mlx_paged_attn::ColdSidecarPolicy {
    mlx_paged_attn::ColdSidecarPolicy::new(mlx_paged_attn::ColdSidecarLayout {
        group: mlx_paged_attn::ColdGroup::GdnState,
        boundary_tokens: 0,
        num_layers: 4,
        tensors_per_layer: 2,
        dtype: "BFloat16".to_string(),
        dims: vec![1, 8, 128],
        bytes_per_tensor: 8 * 128 * 2,
    })
    .expect("a GdnState sidecar policy must validate")
}

/// The master switch, executed for real.
///
/// `gemma4_sliding_cold_ladder_wanted` decides FOUR things — whether the
/// prefill publishes rungs, whether a stored entry is FLAGGED a rung,
/// whether decode publishes off-cadence, and whether the ladder byte cap
/// runs — and until now nothing ran it. Every test built its caps by handing
/// `gemma4_sliding_retention_caps_for_override` an explicit boolean, so
/// making this predicate return `false` unconditionally left the cold tier
/// completely inert with the whole suite green.
///
/// What this cannot reach is the `paged_adapter -> cold_tier()` borrow in
/// `Gemma4Inner::gemma4_sliding_retention_caps_for_turn`, which needs a
/// constructed adapter (Metal) and a loaded checkpoint;
/// `paged_kv_cache_adapter::tests::cold_tier_defaults_none_and_holds_context_across_resets`
/// pins that accessor, and
/// `gemma4_sliding_ladder_intent_has_one_production_source` pins that the
/// borrow is the ONLY thing that call site contributes.
#[test]
fn gemma4_sliding_cold_ladder_wants_a_ladder_only_for_a_sliding_window_sidecar() {
    let config = twelve_b_sliding_config();
    let block_size = 16u32;

    assert!(
        !gemma4_sliding_cold_ladder_wanted(None),
        "no cold tier: nothing could ever read a rung"
    );

    let (no_policy, root_no_policy) = cold_tier_context_with("no-policy", None);
    assert!(
        !gemma4_sliding_cold_ladder_wanted(Some(&no_policy)),
        "a family whose whole per-token state lives in the paged pool (dense qwen3) \
         installs no sidecar policy, so it has no auxiliary state to anchor"
    );

    let (gdn, root_gdn) = cold_tier_context_with("gdn", Some(gdn_sidecar_policy()));
    assert!(
        !gemma4_sliding_cold_ladder_wanted(Some(&gdn)),
        "another family's sidecar group must not switch gemma4's ladder on"
    );

    let policy = sliding_sidecar::policy(&config).expect("the 12B geometry has a policy");
    assert_eq!(policy.group(), mlx_paged_attn::ColdGroup::SlidingWindow);
    let (sliding, root_sliding) = cold_tier_context_with("sliding", Some(policy));
    assert!(
        gemma4_sliding_cold_ladder_wanted(Some(&sliding)),
        "THE persist turn: a SlidingWindow sidecar policy is exactly what \
         capture_gemma4_sliding_cold_sidecar needs a rung for"
    );

    // And the caps every publish/retention seam reads follow it, with no
    // second opinion about whether this is a persist turn.
    assert_eq!(
        gemma4_sliding_retention_caps_for_cold_tier(&config, Some(&sliding), block_size),
        gemma4_sliding_retention_caps(&config, block_size, true),
        "a SlidingWindow cold tier must produce the Ladder arm"
    );
    assert_eq!(
        gemma4_sliding_retention_caps_for_cold_tier(&config, None, block_size),
        gemma4_sliding_retention_caps(&config, block_size, false),
        "no cold tier must produce the pre-ladder arm verbatim"
    );
    assert!(
        gemma4_sliding_retention_caps_for_cold_tier(&config, Some(&sliding), block_size)
            .wants_ladder()
    );
    assert!(
        !gemma4_sliding_retention_caps_for_cold_tier(&config, Some(&gdn), block_size)
            .wants_ladder()
    );

    for root in [root_no_policy, root_gdn, root_sliding] {
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// The decode publisher's DECISION, driven through production's own caps
/// derivation rather than through the free predicate.
///
/// Both decode tests above call `gemma4_sliding_decode_publishes_checkpoint`
/// directly with hand-built caps, so hard-coding `want_ladder` to `false`
/// inside the decode publisher reverted it to cadence-only — defect B fully
/// un-fixed in production — with both of them still green. This one starts
/// from a `ColdTierContext`, which is what the adapter actually hands over.
#[test]
fn gemma4_sliding_decode_boundary_plan_reads_the_turns_real_cold_tier() {
    let config = twelve_b_sliding_config();
    let block_size = 16u32;
    let policy = sliding_sidecar::policy(&config).expect("the 12B geometry has a policy");
    let (persisting, root) = cold_tier_context_with("decode-plan", Some(policy));

    // 256 is a rung and NOT a cadence multiple (the cadence is one whole
    // window, 1024). It is published by decode or by nothing at all.
    assert_eq!(
        gemma4_sliding_decode_boundary_plan(&config, Some(&persisting), block_size, 256),
        Some(Gemma4SlidingDecodeBoundary {
            prefix_len: 256,
            block_size,
            checkpoint_interval: 1024,
            on_anchor_rung: true,
        }),
        "a persist turn must publish the sub-window rung the cadence skips"
    );
    assert_eq!(
        gemma4_sliding_decode_boundary_plan(&config, None, block_size, 256),
        None,
        "persistence-OFF keeps the bare cadence: the same cursor publishes nothing"
    );

    for (label, cold) in [("persist", Some(&persisting)), ("off", None)] {
        assert_eq!(
            gemma4_sliding_decode_boundary_plan(&config, cold, block_size, 1024),
            Some(Gemma4SlidingDecodeBoundary {
                prefix_len: 1024,
                block_size,
                checkpoint_interval: 1024,
                on_anchor_rung: cold.is_some(),
            }),
            "{label}: the cadence fires on both arms; only the rung FLAG differs"
        );
        assert_eq!(
            gemma4_sliding_decode_boundary_plan(&config, cold, block_size, 300),
            None,
            "{label}: an ordinary decode step publishes nothing"
        );
        assert_eq!(
            gemma4_sliding_decode_boundary_plan(&config, cold, block_size, 0),
            None,
            "{label}: an empty request publishes nothing"
        );
    }

    let _ = std::fs::remove_dir_all(&root);
}

/// The decode publisher now tests the cheap cadence/grid arithmetic BEFORE
/// deriving caps, because HEAD returned early on every non-boundary step
/// and deriving caps walks `num_hidden_layers` three times over (96
/// `String == "full_attention"` compares on the 12B) plus an env `OnceLock`
/// read — per decode token, on every gemma4 paged turn, persistence-OFF
/// included.
///
/// A short-circuit that changes WHICH cursors publish would change emitted
/// tokens, so pin the two against each other on every cursor across four
/// windows, on both arms.
#[test]
fn gemma4_sliding_decode_plan_matches_the_publish_predicate_on_every_cursor() {
    let config = twelve_b_sliding_config();
    let block_size = 16u32;
    let policy = sliding_sidecar::policy(&config).expect("the 12B geometry has a policy");
    let (persisting, root) = cold_tier_context_with("decode-equiv", Some(policy));
    let interval = gemma4_sliding_decode_checkpoint_interval(&config, block_size);

    for (label, cold, want_ladder) in [("persist", Some(&persisting), true), ("off", None, false)] {
        let caps = gemma4_sliding_retention_caps(&config, block_size, want_ladder);
        let mut published = Vec::new();
        for cursor in 0..=4200u32 {
            let planned = gemma4_sliding_decode_boundary_plan(&config, cold, block_size, cursor);
            assert_eq!(
                planned.is_some(),
                gemma4_sliding_decode_publishes_checkpoint(cursor, interval, caps),
                "{label}: cursor {cursor} disagrees with the publish predicate"
            );
            if planned.is_some() {
                published.push(cursor);
            }
        }
        let expected: Vec<u32> = if want_ladder {
            vec![64, 256, 1024, 2048, 3072, 4096]
        } else {
            vec![1024, 2048, 3072, 4096]
        };
        assert_eq!(
            published, expected,
            "{label}: the short-circuit must not move the published set"
        );
    }

    let _ = std::fs::remove_dir_all(&root);
}

/// The anchor-grid pre-test is only allowed to be CHEAP, never selective:
/// it must accept every boundary any published rung set could contain, or
/// the decode publisher silently drops rungs on some geometry.
#[test]
fn gemma4_sliding_anchor_grid_pretest_is_a_superset_of_every_published_rung() {
    for (label, config) in [
        ("12B", twelve_b_sliding_config()),
        ("kv-shared", kv_shared_sliding_config()),
        ("narrow-window", narrow_window_sliding_config()),
        ("all-global", all_global_config()),
    ] {
        for block_size in [8u32, 16, 32] {
            for base_limit in [1usize, 2, 6] {
                for rung in gemma4_sliding_cold_anchor_rungs(&config, block_size, base_limit) {
                    assert!(
                        gemma4_sliding_prefix_len_is_on_the_anchor_grid(rung, block_size),
                        "{label}: published rung {rung} (block {block_size}, base limit \
                         {base_limit}) is not on the grid the decode fast path screens with"
                    );
                }
            }
        }
    }
    // ...and it is genuinely a screen, not `true`.
    assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(300, 16));
    assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(16, 16));
    assert!(
        !gemma4_sliding_prefix_len_is_on_the_anchor_grid(16384, 16),
        "the grid stops at GEMMA4_SLIDING_ANCHOR_MAX_RUNGS"
    );
    assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(64, 0));
    assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(0, 16));
}

/// Production text of the family: the hub plus every seam it declares, in
/// hub-first declaration order. There is no test text to trim any more — the
/// unit-test modules are files of their own and none of them is a member here
/// — so `gemma4_sliding_ladder_intent_has_one_production_source` asserts that
/// separation instead of splitting on a header literal.
static SOURCE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    [
        include_str!("../model.rs"),
        include_str!("kv_cache.rs"),
        include_str!("stream.rs"),
        include_str!("sliding.rs"),
        include_str!("construct.rs"),
        include_str!("sliding_cold.rs"),
        include_str!("multimodal.rs"),
        include_str!("paged_forward.rs"),
        include_str!("backend_impl.rs"),
        include_str!("forward.rs"),
    ]
    .join("\n")
});

fn production_source() -> &'static str {
    &SOURCE
}

/// Production lines, trimmed, with line comments and `[MLX_TRACE]` format
/// strings dropped. Doc comments quote these identifiers constantly and a
/// trace line prints `cold_anchor_rung={}` / names the caps helpers in
/// prose; neither is code, and counting them makes every guard below a
/// comment-editing tripwire instead of a structural one.
fn production_code_lines() -> Vec<&'static str> {
    production_source()
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("//") && !line.contains("[MLX_TRACE]"))
        .collect()
}

/// Every production line that could WRITE `cold_anchor_rung`: the field name
/// followed, after any spacing, by `:` (a field declaration or a struct
/// literal) or by a LONE `=` (an assignment). `==` and `!=` are reads, and
/// so is a bare `checkpoint.cold_anchor_rung` in a predicate.
///
/// The `=` arm is the whole reason this is not a substring count. The guard
/// this replaces was `matches("cold_anchor_rung:").count() == 2`, which sees
/// struct-literal syntax only — so the shortest way to restore the defect it
/// exists to prevent slipped straight past it:
///
/// ```text
///   if let Some(last) = self.sliding_prefix_checkpoints.back_mut() {
///       last.cold_anchor_rung = false;      // no colon: count stays 2
///   }
/// ```
fn cold_anchor_rung_write_sites() -> Vec<&'static str> {
    production_code_lines()
        .into_iter()
        .filter(|line| {
            line.match_indices("cold_anchor_rung")
                .any(|(index, needle)| {
                    // A `&mut` borrow hands the field to someone else to
                    // write — `mem::take`/`replace`/`swap` set it without
                    // ever naming an operator here. The borrow reaches the
                    // field through a path (`&mut last.cold_anchor_rung`),
                    // so accept anything between the `&mut` and the field
                    // that is still part of that path.
                    let before = &line[..index];
                    if let Some(borrow) = before.rfind("&mut") {
                        let path = &before[borrow + "&mut".len()..];
                        if path
                            .chars()
                            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | ':' | ' '))
                        {
                            return true;
                        }
                    }
                    let mut rest = line[index + needle.len()..].trim_start().chars();
                    match rest.next() {
                        Some(':') => true,
                        // Plain assignment, but NOT the `==` comparison.
                        Some('=') => rest.next() != Some('='),
                        // Compound assignment. `x &= flag` is the worst of
                        // these: with a value that is false on a normal
                        // turn it clears the flag on every stored entry and
                        // restores the defect this guard exists to prevent,
                        // while naming no `=` of its own.
                        Some('&' | '|' | '^' | '+' | '-' | '*' | '/' | '%') => {
                            rest.next() == Some('=')
                        }
                        _ => false,
                    }
                })
        })
        .collect()
}

/// `cold_anchor_rung` is the ladder's whole eviction ordering, and d134ab3e
/// shipped four visually identical `cold_anchor_rung: false` literals of
/// which two were load-bearing and two were dead. A test cannot execute the
/// publish sites without a GPU and real caches, so what is pinned here is
/// the structure that makes the derivation unforgeable: production WRITES
/// the field in exactly two places — the declaration, and the single
/// derivation in `Gemma4SlidingPrefixCheckpointDraft::into_checkpoint`.
/// Every publish site hands over a draft, which has no such field.
///
/// Two things this cannot see, both covered elsewhere:
///   * a publish site handing `into_checkpoint` the WRONG caps (a literal
///     `Gemma4SlidingRetentionCaps::pre_ladder(..)` instead of the turn's) —
///     `gemma4_sliding_ladder_intent_has_one_production_source` pins the
///     construction sites of the caps themselves;
///   * a rung that is never PUBLISHED at all, which is
///     `gemma4_sliding_decode_boundary_plan`'s job and is tested against the
///     real cold-tier derivation.
#[test]
fn gemma4_sliding_anchor_flag_has_exactly_one_writer() {
    assert_eq!(
        cold_anchor_rung_write_sites(),
        vec![
            "cold_anchor_rung: bool,",
            "cold_anchor_rung: caps.wants_ladder() && caps.anchors.contains(self.prefix_len),",
        ],
        "expected the field declaration plus `into_checkpoint`'s derivation and \
         nothing else. A publish site that sets the flag by hand — in struct-literal \
         OR assignment form — can clear it on a real rung, and an unflagged rung is \
         the ladder's PREFERRED eviction victim, i.e. born then immediately evicted"
    );
}

/// The production wiring the ladder hangs off, pinned as text because the
/// two call sites that consume it are only reachable with a GPU and a
/// loaded checkpoint.
///
/// Both of these mutations disconnect the feature in production and neither
/// changed a single behavioural test:
///
/// ```text
///   prefill orchestrator:  self.gemma4_sliding_retention_caps_for_turn(block_size)
///                       -> gemma4_sliding_retention_caps(&config, block_size, false)
///   decode publisher:      the same substitution
/// ```
///
/// Both work by introducing a SECOND place that picks the `want_ladder`
/// boolean. So what is pinned is that production picks it once: the
/// bool-taking constructors are called exactly where they are defined to be
/// called, and the only producer of the boolean is
/// `gemma4_sliding_cold_ladder_wanted`, whose behaviour
/// `gemma4_sliding_cold_ladder_wants_a_ladder_only_for_a_sliding_window_sidecar`
/// tests for real.
///
/// Counts include each function's own definition, so "2" reads as
/// "defined once, called once".
#[test]
fn gemma4_sliding_ladder_intent_has_one_production_source() {
    assert!(
        !production_source().contains("fn cold_anchor_rung_write_sites"),
        "a test file leaked into the production source list"
    );
    // Every production seam the hub declares must be a member of `SOURCE`, or
    // its lines vanish from every count below without failing anything. Only
    // the `#[cfg(test)]` children and the `#[path]`-redirected scheduler are
    // legitimately absent.
    let hub: Vec<&str> = include_str!("../model.rs").lines().collect();
    let declared: Vec<&str> = hub
        .iter()
        .enumerate()
        .filter(|(index, line)| {
            line.starts_with("mod ")
                && *index > 0
                && hub[index - 1] != "#[cfg(test)]"
                && !hub[index - 1].starts_with("#[path")
        })
        .filter_map(|(_, line)| line.strip_prefix("mod ")?.strip_suffix(';'))
        .collect();
    assert_eq!(
        declared.len(),
        9,
        "the include_str! member list drifted from the hub's mod block: {declared:?}"
    );
    let code = production_code_lines().join("\n");
    for (needle, expected, why) in [
        (
            "gemma4_sliding_cold_ladder_wanted(",
            4usize,
            "defined once; called from `gemma4_sliding_retention_caps_for_cold_tier` (the \
             derivation), `gemma4_sliding_decode_boundary_plan`'s hot-path screen, and the \
             grouped sliding checkpoint publisher. The screen is covered behaviourally by \
             `gemma4_sliding_decode_boundary_plan_reads_the_turns_real_cold_tier`; the \
             publisher is covered by the grouped cold-tier restart parity test",
        ),
        (
            "gemma4_sliding_retention_caps(",
            2,
            "defined once, called once (from `gemma4_sliding_retention_caps_for_cold_tier`); \
             a second call is how both disconnect mutations spell themselves",
        ),
        (
            "gemma4_sliding_retention_caps_for_override(",
            2,
            "defined once, called once (from `gemma4_sliding_retention_caps`); production \
             must not reach past the env override",
        ),
        (
            "gemma4_sliding_retention_caps_for_cold_tier(",
            6,
            "defined once, called from the grouped checkpoint publisher, grouped finalize's \
             live-anchor fallback, the scheduler's anchor query, \
             `gemma4_sliding_retention_caps_for_turn`, and \
             `gemma4_sliding_decode_boundary_plan`",
        ),
        (
            "Gemma4SlidingRetentionCaps::pre_ladder(",
            1,
            "only `gemma4_sliding_retention_caps_for_override` may build the OFF arm; a \
             publish site building one by hand would hand `into_checkpoint` caps that \
             clear the anchor flag on a genuine rung",
        ),
        (
            "Gemma4SlidingRetentionCaps::ladder(",
            2,
            "only `gemma4_sliding_retention_caps_for_override`'s two return sites (the \
             operator-override arm and the widened arm)",
        ),
        (
            "cold_tier()",
            5,
            "the grouped publisher, scheduler anchor query, grouped sidecar capture \
             metadata, grouped sidecar enqueue, and \
             `gemma4_sliding_retention_caps_for_turn` each borrow the live cold-tier \
             context. Passing a literal `None` at any decision/capture seam disconnects \
             persistence while leaving the pure derivation untouched",
        ),
    ] {
        assert_eq!(
            code.matches(needle).count(),
            expected,
            "production mentions `{needle}` {} time(s), expected {expected}: {why}",
            code.matches(needle).count()
        );
    }
}

/// The FLAT draft lane still publishes no sliding decode-boundary
/// checkpoint, and that stays harmless only while it owns no paged state:
/// `capture_gemma4_sliding_cold_sidecar` needs a `PagedKVCacheAdapter`,
/// and an assistant turn runs on `Gemma4LayerCache` arrays with the pools
/// hidden. `assistant_decode.rs` must therefore never reach for the
/// adapter — the day it does, the publisher has to be wired in, and NOT
/// as a copy of the AR call: speculative decode accepts a variable number
/// of tokens per cycle, so the cursor can step from below a rung to above
/// it without ever landing on it.
///
/// The paged DSpark lane already answers this. Its post-commit settle
/// (`Gemma4DsparkStepper::settle_at_committed_frontier` ->
/// `settle_grouped_kv_step_at`) walks `gemma4_cold_rung_candidates`, whose
/// predicate is `boundary <= frontier` rather than `boundary == frontier`,
/// so a jumped rung is still captured — pinned end to end by
/// `family_settle_at_the_committed_frontier_captures_the_cold_rung`.
#[test]
fn the_flat_draft_decode_path_never_touches_the_paged_adapter() {
    let source = include_str!("../assistant_decode.rs");
    // Comments name the field to explain why it is absent; code must not.
    let code = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !code.contains("paged_adapter") && !code.contains("kv_cache_coordinator"),
        "assistant_decode.rs reaches for paged state. The flat lane owning no paged \
         state is the only reason it is safe for it not to publish a sliding \
         decode-boundary checkpoint. If that has changed, wire the publisher in — \
         with a crossed-boundary predicate, not a landed-on-boundary one: a variable \
         accept count can jump the cursor straight over a rung"
    );
}

/// A DSpark turn's real, production-derived plan: the paged pools are
/// visible, so the plan advertises paged draft support and the turn
/// reaches the PAGED handler, where `run_paged_turn`'s speculative branch
/// drives `run_paged_dspark_turn`.
///
/// This is the positive control for the two edits that enable D1 — the
/// `supports_paged_attention` flip and the scheduler's narrowed flat-lane
/// predicate. Either one alone is not enough: with the pools hidden
/// (`install_flat_owner_caches`) a DSpark request advertises no
/// speculative plan at all and downgrades to AR, which the second half of
/// this test pins.
#[test]
fn gemma4_paged_dspark_plan_routes_to_the_paged_handler() {
    let Some(inner) = crate::models::gemma4::dspark_decode::tests::tiny_paged_inner_with_draft()
    else {
        eprintln!("skipping: this build cannot back the paged KV pools");
        return;
    };
    let execution = inner.execution_plan();
    assert!(
        execution.paged_attention.is_some(),
        "a DSpark command runs on the paged lane, so the pools stay visible"
    );
    let speculative = match execution.speculative {
        Some(s) => s,
        None => panic!("a loaded DSpark draft must be advertised on the paged lane"),
    };
    assert!(
        speculative.supports_paged_attention,
        "DSpark proposal/verification is implemented against the paged pools"
    );

    let request = TurnRequest {
        is_delta: false,
        input_media: MediaCapabilities::NONE,
        context_media: MediaCapabilities::NONE,
        speculative_requested: true,
        streaming: false,
    };
    let plan = TurnPlan::resolve(execution, request);
    assert_eq!(
        plan.decoder,
        DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        "paged + DSpark + opt-in must select the draft decoder"
    );
    assert_eq!(
        plan.path(),
        TurnPath::Paged,
        "`path()` checks paged first; `run_paged_turn`'s speculative branch \
         owns the turn from there"
    );

    // Hiding the pools does not silently downgrade to a flat DSpark turn —
    // there is none. The plan drops speculation entirely and says so.
    let mut hidden = inner;
    hidden.install_flat_owner_caches(None);
    let hidden_execution = hidden.execution_plan();
    assert!(
        hidden_execution.speculative.is_none(),
        "with the pools hidden a DSpark draft has no lane, so none is offered"
    );
    assert_eq!(
        TurnPlan::resolve(hidden_execution, request).decoder,
        DecoderPlan::Autoregressive,
        "a laneless DSpark request downgrades to exact AR rather than dropping"
    );
    assert!(
        hidden.kv_cache_coordinator.is_some(),
        "hiding the pools must not destroy them"
    );
}

/// The ASSISTANT drafter keeps the flat lane: its Q-only attention reads
/// the target's flat `Gemma4LayerCache` K/V directly, so its plan declares
/// no paged support and the turn reaches the flat speculative handler.
#[test]
fn gemma4_assistant_draft_plan_routes_to_the_flat_speculative_handler() {
    let inner = crate::models::gemma4::dspark_decode::tests::tiny_inner_with_assistant_draft();
    let execution = inner.execution_plan();

    assert!(
        execution.paged_attention.is_none(),
        "the flat assistant lane must hide paged attention from the turn planner"
    );
    let speculative = match execution.speculative {
        Some(s) => s,
        None => panic!("tiny_inner_with_assistant_draft carries a draft; advertise it"),
    };
    assert!(
        !speculative.supports_paged_attention,
        "the assistant drafter has no paged proposal/verification path"
    );

    let plan = TurnPlan::resolve(
        execution,
        TurnRequest {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            speculative_requested: true,
            streaming: false,
        },
    );
    assert_eq!(
        plan.decoder,
        DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        "flat + assistant draft + opt-in must select the draft decoder"
    );
    assert_eq!(
        plan.path(),
        TurnPath::Speculative,
        "the assistant decoder runs through `run_speculative_turn`"
    );
}

/// One PAGED owner serves media turns and speculative turns in the same
/// session, and the media state decides which decoder it gets: while the
/// live prefix carries media the speculative plan does not cover it and
/// the turn downgrades to exact AR on the SAME owner; once a fresh
/// text turn replaces that context, speculation is admitted again — no
/// lane switch, no second owner, and the coordinator is never rebuilt.
///
/// Mutation this catches: widening the DSpark plan's
/// `supported_context_media`, which would run speculation over a media
/// prefix the drafter's tapped context never saw.
#[test]
fn gemma4_paged_owner_serves_media_then_speculative_turns() {
    let Some(mut inner) =
        crate::models::gemma4::dspark_decode::tests::tiny_paged_inner_with_draft()
    else {
        eprintln!("skipping: this build cannot back the paged KV pools");
        return;
    };
    inner.set_active_paged_owner(21);
    let coordinator_groups = inner
        .kv_cache_coordinator
        .as_ref()
        .map(|coordinator| coordinator.routes().len());

    // A live media prefix: speculation is offered but does not cover the
    // context, so the planner downgrades this turn only.
    inner.media_session_context = MediaCapabilities::IMAGES;
    let continued = TurnPlan::resolve(
        inner.execution_plan(),
        TurnRequest {
            is_delta: true,
            input_media: MediaCapabilities::NONE,
            context_media: inner.media_session_context,
            speculative_requested: true,
            streaming: false,
        },
    );
    assert_eq!(
        continued.decoder,
        DecoderPlan::Autoregressive,
        "a speculative request over a media prefix must downgrade, not run"
    );
    assert_eq!(
        continued.path(),
        TurnPath::Paged,
        "the downgraded turn stays on the same paged owner"
    );

    // A fresh text turn replaces the media context; the same owner now
    // takes the speculative decoder.
    inner.media_session_context = MediaCapabilities::NONE;
    let fresh = TurnPlan::resolve(
        inner.execution_plan(),
        TurnRequest {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            speculative_requested: true,
            streaming: false,
        },
    );
    assert_eq!(
        fresh.decoder,
        DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        "a text-only prefix on the same owner admits speculation"
    );
    assert_eq!(fresh.path(), TurnPath::Paged);
    assert_eq!(
        inner.active_paged_seq, 21,
        "both turns run on the one owner the scheduler selected"
    );
    assert_eq!(
        inner
            .kv_cache_coordinator
            .as_ref()
            .map(|coordinator| coordinator.routes().len()),
        coordinator_groups,
        "neither turn may rebuild the grouped coordinator"
    );
}

/// The one lane that still installs flat caches — the ASSISTANT drafter —
/// must HIDE the resident paged pools for its command, not destroy them:
/// the very next AR command on the same loaded model runs paged.
#[test]
fn gemma4_resident_paged_pools_are_hidden_only_for_the_assistant_lane() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    let mut draft_fixture =
        crate::models::gemma4::dspark_decode::tests::tiny_inner_with_assistant_draft();
    let mut inner = Gemma4Inner::new(paged_tiny_config(Some(true)))
        .expect("construct tiny paged Gemma4 target");
    inner.draft = draft_fixture.draft.take();
    assert!(inner.kv_cache_coordinator.is_some());

    inner.install_flat_owner_caches(None);
    let draft_plan = TurnPlan::resolve(
        inner.execution_plan(),
        TurnRequest {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            speculative_requested: true,
            streaming: false,
        },
    );
    assert_eq!(
        draft_plan.path(),
        TurnPath::Speculative,
        "the flat request lane must hide, not destroy, resident paged pools"
    );
    assert!(inner.kv_cache_coordinator.is_some());

    inner.set_active_paged_owner(17);
    let ar_plan = TurnPlan::resolve(
        inner.execution_plan(),
        TurnRequest {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            speculative_requested: false,
            streaming: false,
        },
    );
    assert_eq!(ar_plan.path(), TurnPath::Paged);
}

#[test]
fn gemma4_grouped_pool_accounting_covers_every_private_adapter() {
    let inner = Gemma4Inner::new(paged_tiny_config(Some(true)))
        .expect("construct tiny paged Gemma4 target");
    let coordinator = inner
        .kv_cache_coordinator
        .as_ref()
        .expect("paged target must own a grouped coordinator");
    let expected = (0..coordinator.inner.groups().len())
        .map(|group_id| {
            coordinator
                .adapter(group_id)
                .and_then(PagedKVCacheAdapter::pool_allocated_bytes)
        })
        .try_fold(0u64, |total, bytes| {
            bytes.map(|bytes| total.saturating_add(bytes))
        })
        .expect("read every grouped pool footprint");
    assert!(expected > 0, "paged Gemma4 must allocate private KV pools");
    assert_eq!(
        coordinator
            .pool_allocated_bytes()
            .expect("report grouped pool footprint"),
        expected
    );
}

/// A finished conversation's anchors must not squat. `Ladder` defers
/// anchors, it does not protect them: once no non-anchor is left, the
/// first anchor that is NOT an ancestor of the newest entry goes.
///
/// The interleaving is what makes this discriminating, and it is the
/// interleaving several conversations multiplexed over one model actually
/// produce. `B@64` is pushed FIRST, so a victim rule that only walks the
/// deque in publish order takes it — even though it is a strict ancestor
/// of the entry being published right now, i.e. the single most reusable
/// thing in the store. Only the ancestor test skips past it to `A@64`.
///
/// ```text
///   store (oldest first)   B@64   A@64   A@256   <- push B@256, limit 3
///   with the ancestor test        ----          evicted: A@64
///   publish order only     ----                 evicted: B@64   (wrong)
/// ```
#[test]
fn gemma4_sliding_ladder_evicts_a_stale_lineage_anchor() {
    let block_size = 16;
    let caps = Gemma4SlidingRetentionCaps::ladder(
        3,
        Gemma4SlidingAnchorRungs::from_slice(&[64, 256, 1024]),
        Gemma4SlidingCheckpointBytes::for_config(&twelve_b_sliding_config()),
    );
    let lineage_a: Vec<u32> = (0..4096).collect();
    let lineage_b: Vec<u32> = (0..4096).map(|token| token + 90_000).collect();
    let from_b = |checkpoint: &Gemma4SlidingPrefixCheckpoint| {
        checkpoint.tokens.first().copied().unwrap_or(0) >= 90_000
    };

    let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
    for (rung, tokens) in [
        (64u32, &lineage_b),
        (64, &lineage_a),
        (256, &lineage_a),
        (256, &lineage_b),
    ] {
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut retained,
            sliding_checkpoint_at(rung, block_size, tokens),
            caps,
            false,
        );
    }

    let survivors: Vec<(u32, bool)> = retained
        .iter()
        .map(|checkpoint| (checkpoint.prefix_len, from_b(checkpoint)))
        .collect();
    assert_eq!(
        survivors,
        vec![(64, true), (256, false), (256, true)],
        "the stale lineage's anchor must go, not the newest entry's own ancestor"
    );
}

#[test]
fn gemma4_prompt_boundary_retains_a_across_a_b_a_image_identity() {
    let tokens: Vec<u32> = (1..=12).collect();
    let block_size = 4;
    let prefix_len = 8;
    let a_keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[(4, 0xAAAA)]);
    let b_keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[(4, 0xBBBB)]);
    let a_hash = compute_gemma4_paged_prefix_block_hash_with_keys(
        &tokens, prefix_len, block_size, &a_keys, 0,
    )
    .expect("A image-aware prefix hash");
    let b_hash = compute_gemma4_paged_prefix_block_hash_with_keys(
        &tokens, prefix_len, block_size, &b_keys, 0,
    )
    .expect("B image-aware prefix hash");
    assert_ne!(a_hash, b_hash);

    let checkpoint = |final_block_hash| Gemma4SlidingPrefixCheckpointDraft {
        prefix_len,
        block_size,
        final_block_hash,
        protected_image_prompt_boundary: true,
        tokens: tokens[..prefix_len as usize].to_vec(),
        snapshots: Vec::new(),
    };
    let bytes = Gemma4SlidingCheckpointBytes::for_config(&twelve_b_sliding_config());
    let mut retained = VecDeque::new();
    upsert_gemma4_sliding_prefix_checkpoint(
        &mut retained,
        checkpoint(a_hash),
        Gemma4SlidingRetentionCaps::pre_ladder(8, bytes),
        false,
    );
    let mut latest_prompt_boundary = checkpoint(a_hash);
    assert_eq!(latest_prompt_boundary.final_block_hash, a_hash);
    upsert_gemma4_sliding_prefix_checkpoint(
        &mut retained,
        checkpoint(b_hash),
        Gemma4SlidingRetentionCaps::pre_ladder(8, bytes),
        false,
    );
    latest_prompt_boundary = checkpoint(b_hash);

    assert_eq!(latest_prompt_boundary.final_block_hash, b_hash);
    assert!(
        retained
            .iter()
            .any(|entry| entry.final_block_hash == a_hash)
    );
    assert!(
        retained
            .iter()
            .any(|entry| entry.final_block_hash == b_hash)
    );
    upsert_gemma4_sliding_prefix_checkpoint(
        &mut retained,
        Gemma4SlidingPrefixCheckpointDraft {
            prefix_len,
            block_size,
            final_block_hash: 0xDEC0DE,
            protected_image_prompt_boundary: false,
            tokens: tokens[..prefix_len as usize].to_vec(),
            snapshots: Vec::new(),
        },
        Gemma4SlidingRetentionCaps::pre_ladder(2, bytes),
        false,
    );
    assert_eq!(retained.len(), 2);
    assert!(
        retained
            .iter()
            .any(|entry| entry.final_block_hash == a_hash),
        "decode checkpoints must not evict protected image A"
    );
    assert!(
        retained
            .iter()
            .any(|entry| entry.final_block_hash == b_hash),
        "decode checkpoints must not evict protected image B"
    );
    let restored_a = retained.iter().rev().find(|entry| {
        entry.prefix_len == prefix_len
            && entry.tokens == tokens[..prefix_len as usize]
            && entry.final_block_hash == a_hash
    });
    assert!(
        restored_a.is_some(),
        "A must remain restorable after B replaces the latest singleton boundary"
    );
}

#[test]
fn image_expansion_requires_template_placeholder() {
    let image_token_id = 258880u32;
    let boi = 255999u32;
    let eoi = 258882u32;
    let image = ProcessedGemma4Image {
        pixel_values: MxArray::zeros(&[1, 1], Some(DType::Float32)).unwrap(),
        num_soft_tokens: 3,
        position_ids: None,
    };
    let error = expand_image_tokens(
        &[2, 9],
        std::slice::from_ref(&image),
        image_token_id,
        boi,
        eoi,
    )
    .expect_err("missing checkpoint-template image placeholder must fail");
    assert!(error.to_string().contains("0 image placeholder(s)"));
}

#[test]
fn stream_dispatch_promotes_channel_only_output_to_visible_text() {
    let (tx, mut rx) = crate::model_thread::stream_channel(8);
    let sender = StreamSender(&tx);
    let mut state = Gemma4StreamDispatchState::default();

    state.dispatch_segments(
        vec![StreamSegment::Reasoning("final answer".into())],
        &sender,
    );
    assert!(rx.try_recv().is_err());

    state.finish(&sender);
    let chunk = rx.try_recv().unwrap().unwrap();
    assert_eq!(chunk.text, "final answer");
    assert_eq!(chunk.is_reasoning, Some(false));
    assert!(rx.try_recv().is_err());
}

#[test]
fn stream_dispatch_keeps_truncated_prompted_channel_as_reasoning() {
    let (tx, mut rx) = crate::model_thread::stream_channel(8);
    let sender = StreamSender(&tx);
    let mut state = Gemma4StreamDispatchState::new(true);

    state.dispatch_segments(
        vec![StreamSegment::Reasoning("unfinished plan".into())],
        &sender,
    );
    assert!(rx.try_recv().is_err());

    state.finish(&sender);
    let chunk = rx.try_recv().unwrap().unwrap();
    assert_eq!(chunk.text, "unfinished plan");
    assert_eq!(chunk.is_reasoning, Some(true));
    assert!(rx.try_recv().is_err());
}

#[test]
fn stream_dispatch_keeps_reasoning_when_visible_text_follows() {
    let (tx, mut rx) = crate::model_thread::stream_channel(8);
    let sender = StreamSender(&tx);
    let mut state = Gemma4StreamDispatchState::default();

    state.dispatch_segments(
        vec![
            StreamSegment::Reasoning("scratch".into()),
            StreamSegment::Text("answer".into()),
        ],
        &sender,
    );
    state.finish(&sender);

    let reasoning = rx.try_recv().unwrap().unwrap();
    assert_eq!(reasoning.text, "scratch");
    assert_eq!(reasoning.is_reasoning, Some(true));

    let text = rx.try_recv().unwrap().unwrap();
    assert_eq!(text.text, "answer");
    assert_eq!(text.is_reasoning, Some(false));
    assert!(rx.try_recv().is_err());
}

#[test]
fn stream_dispatch_keeps_reasoning_when_tool_call_follows() {
    let (tx, mut rx) = crate::model_thread::stream_channel(8);
    let sender = StreamSender(&tx);
    let mut state = Gemma4StreamDispatchState::default();

    state.dispatch_segments(
        vec![
            StreamSegment::Reasoning("scratch".into()),
            StreamSegment::ToolCall,
        ],
        &sender,
    );
    state.finish(&sender);

    let reasoning = rx.try_recv().unwrap().unwrap();
    assert_eq!(reasoning.text, "scratch");
    assert_eq!(reasoning.is_reasoning, Some(true));
    assert!(rx.try_recv().is_err());
}

#[test]
fn promote_channel_only_output_moves_thinking_to_text() {
    let mut parsed = parse_gemma4_output("<|channel>thought\nvisible answer<channel|>");
    promote_channel_only_output(&mut parsed, false);

    assert_eq!(parsed.text, "visible answer");
    assert!(parsed.thinking.is_none());
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn seeded_channel_truncation_is_not_promoted_to_visible_text() {
    let mut parsed = crate::models::gemma4::output_parser::parse_gemma4_output_with_open_channel(
        "unfinished plan",
        true,
    );
    promote_channel_only_output(&mut parsed, true);

    assert!(parsed.text.is_empty());
    assert_eq!(parsed.thinking.as_deref(), Some("unfinished plan"));
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn sliding_mask_is_valid_for_bf16_gqa_attention() {
    let q = MxArray::zeros(&[1, 4, 4, 16], Some(DType::BFloat16)).unwrap();
    let k = MxArray::zeros(&[1, 1, 6, 16], Some(DType::BFloat16)).unwrap();
    let v = MxArray::zeros(&[1, 1, 6, 16], Some(DType::BFloat16)).unwrap();
    let mask = create_sliding_mask(4, 2, 3).unwrap();

    assert_eq!(mask.shape_at(0).unwrap(), 1);
    assert_eq!(mask.shape_at(1).unwrap(), 1);
    assert_eq!(mask.shape_at(2).unwrap(), 4);
    assert_eq!(mask.shape_at(3).unwrap(), 6);

    let out = crate::array::scaled_dot_product_attention(&q, &k, &v, 1.0, Some(&mask)).unwrap();
    let values = out.to_float32().unwrap();
    assert_eq!(values.len(), 4 * 4 * 16);
    assert!(values.iter().all(|v| v.is_finite()));
}

#[test]
fn sliding_mask_offset_uses_rotating_window_view() {
    assert_eq!(sliding_mask_offset_for_chunk(512, 16, 1024), None);
    assert_eq!(sliding_mask_offset_for_chunk(512, 528, 1024), Some(528));
    assert_eq!(sliding_mask_offset_for_chunk(512, 43_688, 1024), Some(1024));
    assert_eq!(sliding_mask_offset_for_chunk(2048, 0, 1024), Some(0));
    assert_eq!(sliding_mask_offset_for_chunk(1, 4096, 1024), None);
}

#[test]
fn test_gemma4_paged_prefill_body_chunk_size_honors_configured_size() {
    assert_eq!(
        super::gemma4_paged_prefill_body_chunk_size(4096, 27_938),
        4096
    );
    assert_eq!(
        super::gemma4_paged_prefill_body_chunk_size(512, 27_938),
        512
    );
    assert_eq!(
        super::gemma4_paged_prefill_body_chunk_size(0, 27_938),
        super::GEMMA4_PREFILL_STEP_SIZE as usize
    );
    assert_eq!(
        super::gemma4_paged_prefill_body_chunk_size(0, 127),
        127,
        "the default bound must not pad a short final chunk"
    );
}

#[test]
fn gemma4_grouped_prefill_uses_bounded_default_chunks() {
    assert_eq!(
        super::gemma4_paged_prefill_group_chunk_size(0),
        super::GEMMA4_PREFILL_STEP_SIZE as u32
    );
    assert_eq!(super::gemma4_paged_prefill_group_chunk_size(-1), 512);
    assert_eq!(super::gemma4_paged_prefill_group_chunk_size(2048), 2048);
}

#[test]
fn test_gemma4_paged_prefill_body_chunk_plan_caps_v2_aux() {
    let plan = super::gemma4_paged_prefill_body_chunk_plan(
        8192,
        27_938,
        16,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::ForceVarlen,
    )
    .unwrap();
    assert_eq!(plan.first().unwrap().len, 8192);
    assert!(plan.iter().any(|chunk| chunk.capped_by_v2_aux_limit));

    let mut expected_start = 0usize;
    let mut expected_position = 16u32;
    for chunk in &plan {
        assert_eq!(chunk.start, expected_start);
        assert_eq!(chunk.first_position, expected_position);
        assert!(super::gemma4_paged_prefill_chunk_route_is_aux_safe(
            chunk.len,
            chunk.first_position,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::ForceVarlen,
        ));
        expected_start += chunk.len;
        expected_position += chunk.len as u32;
    }
    assert_eq!(expected_start, 27_938);

    let forced_sdpa = super::gemma4_paged_prefill_body_chunk_plan(
        8192,
        27_938,
        16,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::NonV2,
    )
    .unwrap();
    assert_eq!(forced_sdpa.len(), 4);
    assert_eq!(forced_sdpa[0].len, 8192);
    assert!(
        forced_sdpa
            .iter()
            .all(|chunk| !chunk.capped_by_v2_aux_limit)
    );

    let auto = super::gemma4_paged_prefill_body_chunk_plan(
        8192,
        27_938,
        16,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::Auto,
    )
    .unwrap();
    assert_eq!(auto.len(), 4);
    assert!(
        auto.iter().all(|chunk| !chunk.capped_by_v2_aux_limit),
        "auto must keep full compute chunks when its safe pre-plan selects SDPA"
    );
}

#[test]
fn test_gemma4_sliding_restore_chunk_plan_avoids_singletons() {
    let mut plan = super::gemma4_paged_prefill_body_chunk_plan(
        4,
        9,
        0,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::NonV2,
    )
    .unwrap();
    assert_eq!(
        plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
        vec![4, 4, 1]
    );

    super::gemma4_coalesce_single_token_restore_chunks(&mut plan);
    assert_eq!(
        plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
        vec![4, 5]
    );
    assert_eq!(plan[1].start, 4);
    assert_eq!(plan[1].first_position, 4);

    let mut one_token_chunks = super::gemma4_paged_prefill_body_chunk_plan(
        1,
        5,
        0,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::NonV2,
    )
    .unwrap();
    super::gemma4_coalesce_single_token_restore_chunks(&mut one_token_chunks);
    assert_eq!(
        one_token_chunks
            .iter()
            .map(|chunk| chunk.len)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(one_token_chunks[1].first_position, 2);
}

#[test]
fn test_gemma4_paged_prefill_chunk_plan_splits_prompt_cache_boundary() {
    let mut plan = super::gemma4_paged_prefill_body_chunk_plan(
        1024,
        1432,
        44_320,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::NonV2,
    )
    .unwrap();
    super::gemma4_split_body_chunk_plan_at_position(&mut plan, 45_744);
    assert_eq!(
        plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
        vec![1024, 400, 8]
    );
    assert_eq!(
        plan.iter()
            .map(|chunk| chunk.first_position)
            .collect::<Vec<_>>(),
        vec![44_320, 45_344, 45_744]
    );
    assert_eq!(
        plan.iter().map(|chunk| chunk.start).collect::<Vec<_>>(),
        vec![0, 1024, 1424]
    );

    let mut unchanged = plan.clone();
    super::gemma4_split_body_chunk_plan_at_position(&mut unchanged, 45_344);
    assert_eq!(unchanged, plan);
}

#[test]
fn test_gemma4_paged_prefill_chunk_plan_is_independent_of_checkpoint_cadence() {
    let plan = super::gemma4_paged_prefill_body_chunk_plan(
        2048,
        3000,
        16,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::NonV2,
    )
    .unwrap();

    assert_eq!(
        plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
        vec![2048, 952]
    );
    assert_eq!(
        plan.iter()
            .map(|chunk| chunk.first_position)
            .collect::<Vec<_>>(),
        vec![16, 2064]
    );
    assert_eq!(
        plan.iter().map(|chunk| chunk.start).collect::<Vec<_>>(),
        vec![0, 2048]
    );

    let capped = super::gemma4_paged_prefill_body_chunk_plan(
        512,
        1600,
        768,
        16,
        1,
        512,
        super::Gemma4PagedPrefillRoutePolicy::NonV2,
    )
    .unwrap();
    assert_eq!(
        capped.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
        vec![512, 512, 512, 64]
    );
    assert!(capped.iter().all(|chunk| chunk.len <= 512));
}

#[test]
fn test_gemma4_sliding_checkpoint_cadence_crosses_unaligned_compute_chunk() {
    assert_eq!(
        super::gemma4_sliding_checkpoint_boundaries_crossed(16, 2064, 1024),
        vec![1024, 2048],
        "a cache hit at 16 followed by one 2K compute chunk must publish both cadence points"
    );
    assert_eq!(
        super::gemma4_sliding_checkpoint_boundaries_crossed(2064, 3016, 1024),
        Vec::<u32>::new()
    );
    assert_eq!(
        super::gemma4_sliding_checkpoint_boundaries_crossed(2064, 4096, 1024),
        vec![3072, 4096]
    );
}

#[test]
fn test_gemma4_sliding_restore_default_is_checkpoint_bounded() {
    let cfg = super::Gemma4Config {
        sliding_window: 1024,
        ..paged_tiny_config(Some(true))
    };

    assert_eq!(
        super::gemma4_large_sliding_restore_suppression_limit_for_override(&cfg, 16, None, 1024),
        None
    );
    assert_eq!(
        super::gemma4_large_sliding_restore_suppression_limit_for_override(&cfg, 16, None, 24_336),
        Some(super::Gemma4SlidingRestoreSuppression {
            limit: 1024,
            source: "default"
        })
    );
}

#[test]
fn test_gemma4_sliding_restore_env_limit_overrides_default() {
    let cfg = super::Gemma4Config {
        sliding_window: 1024,
        ..paged_tiny_config(Some(true))
    };

    assert_eq!(
        super::parse_gemma4_sliding_restore_limit("32768"),
        Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768))
    );
    assert_eq!(
        super::parse_gemma4_sliding_restore_limit(" 44512 "),
        Some(super::Gemma4SlidingRestoreLimitOverride::Cap(44_512))
    );
    assert_eq!(super::parse_gemma4_sliding_restore_limit(""), None);
    assert_eq!(
        super::parse_gemma4_sliding_restore_limit("off"),
        Some(super::Gemma4SlidingRestoreLimitOverride::Uncapped)
    );

    assert_eq!(
        super::gemma4_large_sliding_restore_suppression_limit_for_override(
            &cfg,
            16,
            Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768)),
            32_768
        ),
        None
    );
    assert_eq!(
        super::gemma4_large_sliding_restore_suppression_limit_for_override(
            &cfg,
            16,
            Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768)),
            44_512
        ),
        Some(super::Gemma4SlidingRestoreSuppression {
            limit: 32_768,
            source: "env"
        })
    );
    assert_eq!(
        super::gemma4_large_sliding_restore_suppression_limit_for_override(
            &cfg,
            16,
            Some(super::Gemma4SlidingRestoreLimitOverride::Uncapped),
            1_000_000
        ),
        None
    );
}

#[test]
fn test_ple_oov_masking() {
    // Simulate token IDs where some exceed PLE vocab or are negative
    let input_ids = MxArray::from_int32(&[5, 100, 262143, 0, -1], &[1, 5]).unwrap();
    let ple_vocab = 262144i32; // PLE vocab size

    let ple_vocab_arr = MxArray::scalar_int(ple_vocab).unwrap();
    let zero = MxArray::scalar_int(0).unwrap();
    let valid_mask = input_ids
        .greater_equal(&zero)
        .unwrap()
        .logical_and(&input_ids.less(&ple_vocab_arr).unwrap())
        .unwrap();
    let masked_ids = valid_mask.where_(&input_ids, &zero).unwrap();

    masked_ids.eval();
    // IDs within range: unchanged. IDs out of range (negative): mapped to 0.
    assert_eq!(masked_ids.item_at_int32(0).unwrap(), 5); // in range
    assert_eq!(masked_ids.item_at_int32(1).unwrap(), 100); // in range
    // 262143 < 262144, so it's valid
    assert_eq!(masked_ids.item_at_int32(2).unwrap(), 262143);
    assert_eq!(masked_ids.item_at_int32(3).unwrap(), 0); // in range (0 is valid)
    assert_eq!(masked_ids.item_at_int32(4).unwrap(), 0); // -1 is OOV, mapped to 0
}

/// Tiny Gemma4 config compatible with `LayerKVPool`'s validate
/// constraints (head_size in {32, 64, 96, 128, 256}, FP8 off, etc.).
/// `head_dim = 32`, num_kv_heads = 2, no PLE/MoE/vision/sharing.
#[cfg(test)]
fn paged_tiny_config(use_block_paged: Option<bool>) -> super::Gemma4Config {
    super::Gemma4Config {
        persist_paged_cache: None,
        vocab_size: 100,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        max_position_embeddings: 128,
        sliding_window: 128,
        // All-global so the uniform paged pool's head_dim choice
        // matches every layer trivially.
        layer_types: vec!["full_attention".to_string(), "full_attention".to_string()],
        rope_theta: 1_000_000.0,
        rope_local_base_freq: 10_000.0,
        partial_rotary_factor: 0.25,
        global_num_key_value_heads: None,
        global_head_dim: None,
        attention_k_eq_v: false,
        is_unified: false,
        use_bidirectional_attention: None,
        final_logit_softcapping: None,
        per_layer_input_embeds: false,
        hidden_size_per_layer_input: None,
        vocab_size_per_layer_input: None,
        pad_token_id: 0,
        eos_token_ids: vec![1],
        bos_token_id: 2,
        attention_bias: false,
        use_double_wide_mlp: false,
        num_kv_shared_layers: None,
        default_temperature: None,
        default_top_k: None,
        default_top_p: None,
        enable_moe_block: false,
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
        vision_config: None,
        unified_vision_config: None,
        image_token_id: None,
        boi_token_id: None,
        eoi_token_id: None,
        vision_soft_tokens_per_image: None,
        has_audio: false,
        audio_token_id: None,
        boa_token_id: None,
        eoa_token_id: None,
        audio_samples_per_token: None,
        paged_cache_memory_mb: Some(256),
        paged_block_size: Some(16),
        use_block_paged_cache: use_block_paged,
    }
}

/// `use_block_paged_cache` defaults to `None` when absent from the
/// JSON config — guards against silently switching the storage
/// backend on existing Gemma4 checkpoints.
///
/// Pure-CPU; no MLX runtime needed.
#[test]
fn test_use_block_paged_cache_defaults_to_none_via_serde() {
    let json = serde_json::json!({
        "vocab_size": 0,
        "hidden_size": 0,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "intermediate_size": 1,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
        "max_position_embeddings": 2048,
    });
    let cfg: super::Gemma4Config = serde_json::from_value(json).expect("deserialize Gemma4Config");
    assert_eq!(
        cfg.use_block_paged_cache, None,
        "use_block_paged_cache must default to None on JSON without the key"
    );
    assert_eq!(cfg.paged_block_size, None);
    assert_eq!(cfg.paged_cache_memory_mb, None);
}

/// `use_block_paged_cache: true` round-trips through serde.
#[test]
fn test_use_block_paged_cache_round_trips_true() {
    let json = serde_json::json!({
        "vocab_size": 0,
        "hidden_size": 0,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "intermediate_size": 1,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
        "max_position_embeddings": 2048,
        "use_block_paged_cache": true,
    });
    let cfg: super::Gemma4Config = serde_json::from_value(json).expect("deserialize Gemma4Config");
    assert_eq!(cfg.use_block_paged_cache, Some(true));
}

#[test]
fn test_default_paged_cache_memory_covers_gemma4_full_context() {
    let memory_mb = super::gemma4_default_paged_cache_memory_mb(131_072, 16, 512, 2, 5);
    assert_eq!(
        memory_mb, 2560,
        "Gemma4 26B-A4B global KV cache needs 2560MiB to cover 128k tokens"
    );

    let cfg = mlx_paged_attn::PagedAttentionConfig {
        block_size: 16,
        gpu_memory_mb: memory_mb,
        head_size: 512,
        num_kv_heads: 2,
        num_layers: 5,
        use_fp8_cache: Some(false),
        max_seq_len: Some(131_072),
        max_batch_size: Some(32),
    };
    assert_eq!(cfg.calculate_num_blocks(), 8192);
    assert_eq!(cfg.max_cached_tokens(), 131_072);

    let undersized_cfg = mlx_paged_attn::PagedAttentionConfig {
        gpu_memory_mb: 2048,
        ..cfg
    };
    assert!(
        undersized_cfg.max_cached_tokens() < 124_920,
        "the previous fixed 2048MiB default cannot hold the failed 124,920-token prompt"
    );
}

#[test]
fn test_default_paged_cache_memory_respects_minimum() {
    assert_eq!(
        super::gemma4_default_paged_cache_memory_mb(128, 16, 32, 2, 2),
        256
    );
}

#[test]
fn gemma4_shared_pool_does_not_partition_a_full_sliding_window_per_slot() {
    assert_eq!(
        super::gemma4_group_reserved_blocks(super::AttentionKind::Full, 8192, 8),
        8192
    );
    assert_eq!(
        super::gemma4_group_reserved_blocks(
            super::AttentionKind::SlidingWindow {
                sliding_window: 1024,
            },
            97,
            8,
        ),
        98,
        "one maximum sliding working set plus its null block serves a shared pool"
    );
    assert_eq!(
        super::gemma4_group_reserved_blocks(
            super::AttentionKind::SlidingWindow { sliding_window: 8 },
            2,
            8,
        ),
        9,
        "a tiny window still reserves one starter block per live row plus null"
    );
}

/// Explicit opt-out (`Some(false)`) must NOT allocate the block-paged
/// adapter. The previous "None means no adapter" assertion was removed
/// when the default flipped from `unwrap_or(false)` to `unwrap_or(true)`
/// — the explicit-false path is the new "no adapter" guarantee.
#[test]
fn test_gemma4_inner_no_paged_adapter_when_flag_is_explicit_false() {
    let cfg = paged_tiny_config(Some(false));
    let inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };
    assert!(
        inner.kv_cache_coordinator.is_none(),
        "paged_adapter must be None when use_block_paged_cache is Some(false)"
    );
}

/// Default-flag construction (`None`) must allocate the block-paged
/// adapter under the new default-on policy (`unwrap_or(true)`).
/// Allocates a `LayerKVPool`, so requires Metal — gracefully skips
/// on no-Metal sandboxes.
#[test]
fn test_gemma4_inner_paged_adapter_when_flag_is_none_default_on_macos() {
    // Block-paged needs the Metal backend; on a non-Metal build the
    // adapter is gated off (None) and there is nothing to exercise.
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping (paged backend unavailable without Metal)");
        return;
    }
    let cfg = paged_tiny_config(None);
    match super::Gemma4Inner::new(cfg) {
        Ok(inner) => {
            assert!(
                inner.kv_cache_coordinator.is_some(),
                "paged_adapter must be Some when use_block_paged_cache is None \
                 (new default-on policy: unwrap_or(true))"
            );
        }
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    }
}

/// Construction with `use_block_paged_cache: Some(true)` must populate
/// `paged_adapter`. Allocates a `LayerKVPool`, so requires Metal —
/// gracefully skips on no-Metal sandboxes.
#[test]
fn test_gemma4_inner_constructs_paged_adapter_when_flag_is_true() {
    // Block-paged needs the Metal backend; on a non-Metal build the
    // adapter is gated off (None) and there is nothing to exercise.
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping (paged backend unavailable without Metal)");
        return;
    }
    let cfg = paged_tiny_config(Some(true));
    match super::Gemma4Inner::new(cfg) {
        Ok(inner) => {
            assert!(
                inner.kv_cache_coordinator.is_some(),
                "paged_adapter must be Some when use_block_paged_cache = Some(true)"
            );
        }
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    }
}

/// Paged/flat parity: a fresh (non-reuse) text-only `save_paged_history`
/// must clear `cached_audio_key`, exactly as the flat `save_cache_state`
/// does on a fresh turn. Without that clear, a text-only paged start over a
/// reused model whose prior turn was audio would leave stale media state
/// attached to the replacement session.
#[test]
fn test_text_only_paged_save_clears_stale_audio_key() {
    let cfg = paged_tiny_config(Some(false));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    // Simulate a completed audio turn that left the audio key set, then a
    // fresh text-only paged START (no `reset()`): image key already None,
    // session not continuable.
    inner.cached_audio_key = Some(7);
    inner.cached_image_key = None;
    inner.media_session_continuable = false;

    // Fresh (non-reuse, non-delta) text-only paged save — the same shape
    // the engine uses to persist a fresh text turn's history.
    let save_tokens: Vec<u32> = vec![10, 11, 12];
    let generated: Vec<u32> = vec![20, 21];
    inner
        .save_paged_history(&save_tokens, &generated, false, false)
        .expect("text-only paged save must succeed");

    // The fix: the stale audio key is cleared on the text-only save.
    assert!(
        inner.cached_audio_key.is_none(),
        "text-only paged save must clear the stale audio key"
    );

    assert_eq!(inner.session_media(), MediaCapabilities::NONE);
}

/// A warm text save over a pure-image session extends the same live
/// media-derived KV. It must preserve the raw image identity and ordered
/// placeholder sidecar so later text blocks keep the same image-aware block
/// keys. Audio and mixed turns are deliberately cold/non-continuable and do
/// not enter this path.
#[test]
fn test_media_context_survives_repeated_warm_text_saves() {
    let cfg = paged_tiny_config(Some(false));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    let image_key = 11;
    let image_positions = vec![(1, image_key)];
    inner.publish_media_session_context(Some(image_key), None);
    inner.cached_paged_image_token_positions = image_positions.clone();
    inner.media_session_continuable = true;
    assert_eq!(inner.session_media(), MediaCapabilities::IMAGES);

    for turn in 0..2u32 {
        // Mirrors `run_paged_turn` handing the planner's prior context
        // to `save_paged_history` for a successful warm text delta.
        inner.paged_text_turn_context = inner.session_media();
        inner
            .save_paged_history(&[10, 11, turn], &[20, 21], false, true)
            .expect("warm text paged save must succeed");

        assert_eq!(inner.cached_image_key, Some(image_key));
        assert!(inner.cached_audio_key.is_none());
        assert_eq!(
            inner.cached_paged_image_token_positions, image_positions,
            "turn {turn} must preserve exact image cache lineage"
        );
        assert_eq!(
            inner.session_media(),
            MediaCapabilities::IMAGES,
            "turn {turn} must preserve the image-derived context"
        );
        assert!(inner.media_session_continuable);
    }
}

/// A failed paged media prepare must fail CLOSED. The vision core disarms
/// `media_session_continuable` before the fallible adapter prepare, and all
/// subsequent prepare failures call `invalidate_gemma4_hybrid_session`,
/// which releases the request and clears both global and sliding reuse
/// state. No media-derived history may survive as a token-only prefix hit.
///
/// The state is built with the real transition functions
/// (`publish_media_session_context` → warm `save_paged_history` → marker
/// disarm → `invalidate_gemma4_hybrid_session`); driving the complete
/// multimodal core to the failing prepare needs a real tokenizer file,
/// which unit tests do not have.
#[test]
fn test_failed_media_prepare_fails_closed_after_warm_continuation() {
    let cfg = paged_tiny_config(Some(false));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    // Live caches so the prefix check below can only miss via the media
    // gate (not via `caches.is_none()`).
    inner
        .init_caches_sync()
        .expect("init_caches_sync must succeed");

    // Warm-continued pure-image session: finalize published exact image
    // identity + context and armed the marker; the warm text save preserves
    // that image lineage for later image-aware block registration.
    inner.publish_media_session_context(Some(11), None);
    inner.cached_paged_image_token_positions = vec![(1, 11)];
    inner.media_session_continuable = true;
    inner.paged_text_turn_context = inner.session_media();
    inner
        .save_paged_history(&[100, 101, 102], &[103, 104], false, true)
        .expect("warm text paged save must succeed");
    // Mirrors `run_paged_turn` resetting the turn-scoped snapshot.
    inner.paged_text_turn_context = MediaCapabilities::NONE;
    assert_eq!(inner.cached_image_key, Some(11));
    assert!(inner.cached_audio_key.is_none());
    assert_eq!(inner.cached_paged_image_token_positions, vec![(1, 11)]);
    assert_eq!(inner.session_media(), MediaCapabilities::IMAGES);

    // `keep_all = false` dropped the trailing stop token 104.
    assert_eq!(inner.cached_token_history, vec![100, 101, 102, 103]);
    let delta_tokens: Vec<u32> = vec![100, 101, 102, 103, 200];

    // While the continuation is armed the media gate does not force a
    // prefix miss (warm reuse stays possible).
    assert_eq!(
        inner.verify_cache_prefix(&delta_tokens, true),
        inner.cached_token_history.len(),
        "an armed continuation must not be forced to miss"
    );

    // The next media turn's failure path: the vision core disarms the
    // marker, then its prepare helper invalidates the complete hybrid
    // session before returning the error.
    inner.media_session_continuable = false;
    inner.invalidate_gemma4_hybrid_session("unit-test media prepare failure");

    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_audio_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert_eq!(inner.session_media(), MediaCapabilities::NONE);
    assert!(!inner.media_session_continuable);
    assert_eq!(
        inner.verify_cache_prefix(&delta_tokens, true),
        0,
        "invalidated media history must not seed a text-only prefix hit"
    );
}

#[test]
fn test_fresh_text_save_replaces_persistent_media_context() {
    let cfg = paged_tiny_config(Some(false));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    inner.publish_media_session_context(Some(11), Some(22));
    inner.media_session_continuable = true;
    // Fresh plans carry no prior context. A successful text save therefore
    // replaces, rather than extends, the old media session.
    inner.paged_text_turn_context = MediaCapabilities::NONE;
    inner
        .save_paged_history(&[1, 2, 3], &[4, 5], false, true)
        .expect("fresh text paged save must succeed");

    assert_eq!(inner.session_media(), MediaCapabilities::NONE);
    assert!(!inner.media_session_continuable);
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_audio_key.is_none());
}

/// Image/audio symmetry in `verify_cache_prefix`: a non-continuable session
/// that still holds a cached AUDIO key must MISS (return `0`), exactly as it
/// already does for a cached IMAGE key, so stale media KV is reset instead
/// of being reused as a token-id prefix hit. With an otherwise-hitting
/// prefix (live caches + matching `cached_token_history`), the audio guard
/// must override the would-be hit. A continuable audio session (warm-
/// continue) must NOT be forced to miss by this guard.
///
/// Pre-fix (image-only guard) this would return `cached.len()` for the
/// non-continuable audio case — a HIT — because the audio key was ignored,
/// so the first assertion below would fail.
#[test]
fn test_verify_cache_prefix_audio_key_forces_miss() {
    let cfg = paged_tiny_config(Some(false));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    // Build an otherwise-hitting state: live caches + a non-empty cached
    // history that the incoming tokens match as a prefix. `init_caches_sync`
    // also clears reuse state, so the keys/marker/history are set AFTER.
    inner
        .init_caches_sync()
        .expect("init_caches_sync must succeed");
    inner.cached_token_history = vec![100, 101, 102];
    let tokens: Vec<u32> = vec![100, 101, 102, 103];

    // Non-continuable session holding only an AUDIO key: must MISS.
    inner.cached_image_key = None;
    inner.cached_audio_key = Some(7);
    inner.media_session_continuable = false;
    assert_eq!(
        inner.verify_cache_prefix(&tokens, true),
        0,
        "a non-continuable session holding audio state must force a cache miss"
    );

    // Continuable audio session (warm-continue): the guard must NOT force a
    // miss, so the otherwise-hitting prefix returns `cached.len()`.
    inner.media_session_continuable = true;
    assert_eq!(
        inner.verify_cache_prefix(&tokens, true),
        inner.cached_token_history.len(),
        "a continuable audio session must not be forced to miss by the media guard"
    );

    // Parity check: the same shape with an IMAGE key (already guarded) also
    // misses when non-continuable — the audio branch mirrors it exactly.
    inner.cached_image_key = Some(42);
    inner.cached_audio_key = None;
    inner.media_session_continuable = false;
    assert_eq!(
        inner.verify_cache_prefix(&tokens, true),
        0,
        "a non-continuable session holding image state must force a cache miss"
    );
}

/// Marker reset matrix: `media_session_continuable` must return to `false`
/// at every session-reset entry point so a dropped-media session can never
/// wrongly warm-continue. Covers `clear_reuse_state` and `reset_caches_sync`
/// (both clear via `clear_reuse_state`).
#[test]
fn test_media_session_continuable_reset_matrix() {
    let cfg = paged_tiny_config(Some(false));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    // Fresh construction: marker defaults to false.
    assert!(
        !inner.media_session_continuable,
        "marker must default to false on construction"
    );

    // clear_reuse_state resets the marker and both persistent/transient
    // media context sources.
    inner.publish_media_session_context(Some(7), Some(8));
    inner.paged_text_turn_context = MediaCapabilities::IMAGES_AND_AUDIO;
    inner.media_session_continuable = true;
    inner.clear_reuse_state();
    assert!(
        !inner.media_session_continuable,
        "clear_reuse_state must reset the continuable marker"
    );
    assert_eq!(inner.session_media(), MediaCapabilities::NONE);
    assert_eq!(
        inner.paged_text_turn_context,
        MediaCapabilities::NONE,
        "clear_reuse_state must clear transient turn context"
    );

    // reset_caches_sync (which calls clear_reuse_state) resets the marker
    // AND nulls caches → has_live_session() false → continuation is rejected.
    inner.publish_media_session_context(None, Some(9));
    inner.paged_text_turn_context = MediaCapabilities::AUDIO;
    inner.media_session_continuable = true;
    inner
        .reset_caches_sync()
        .expect("reset_caches_sync must succeed");
    assert!(
        !inner.media_session_continuable,
        "reset_caches_sync must reset the continuable marker"
    );
    assert!(
        inner.cached_audio_key.is_none(),
        "reset_caches_sync must clear the media key"
    );
    assert_eq!(inner.session_media(), MediaCapabilities::NONE);
    assert!(!inner.has_live_session());
}

/// Only pure image turns currently publish image-aware per-block keys.
/// Audio and mixed-media turns stay cold until their non-token identity is
/// represented in the same cache chain.
#[test]
fn test_gemma4_media_continuable_gate() {
    assert!(!gemma4_media_continuable(false, false));
    assert!(gemma4_media_continuable(true, false));
    assert!(!gemma4_media_continuable(false, true));
    assert!(!gemma4_media_continuable(true, true));
}

/// All-global config: every layer must route through `GlobalPaged`
/// with paged_idx == absolute index, no shared layers.
#[test]
fn test_compute_layer_kinds_all_global() {
    let cfg = super::Gemma4Config {
        num_hidden_layers: 4,
        layer_types: vec!["full_attention".to_string(); 4],
        ..paged_tiny_config(None)
    };
    let kinds = super::compute_layer_kinds(&cfg);
    assert_eq!(kinds.len(), 4);
    for (i, k) in kinds.iter().enumerate() {
        match k {
            super::Gemma4LayerKind::GlobalPaged {
                group_id,
                paged_idx,
            } => {
                assert_eq!(*group_id, 0, "all-global layers share group 0");
                assert_eq!(*paged_idx as usize, i, "layer {i} paged_idx mismatch");
            }
            other => panic!("layer {i}: expected GlobalPaged, got {other:?}"),
        }
    }
}

/// Hybrid sliding+global with no sharing: each attention kind gets its
/// own group and physical ordinals are local to that group.
#[test]
fn test_compute_layer_kinds_hybrid_no_sharing() {
    // 5-layer cycle: 4 sliding + 1 global, repeated for 10 layers.
    let cycle = ["sliding_attention"; 4]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once("full_attention".to_string()))
        .collect::<Vec<_>>();
    let layer_types: Vec<String> = (0..10).map(|i| cycle[i % 5].clone()).collect();
    let cfg = super::Gemma4Config {
        num_hidden_layers: 10,
        layer_types,
        ..paged_tiny_config(None)
    };
    let kinds = super::compute_layer_kinds(&cfg);
    // Global layers at indices 4 and 9 -> paged_idx 0, 1.
    for (i, k) in kinds.iter().enumerate() {
        if i == 4 {
            assert!(
                matches!(
                    k,
                    super::Gemma4LayerKind::GlobalPaged {
                        group_id: 0,
                        paged_idx: 0
                    }
                ),
                "layer 4 must be GlobalPaged{{0}}, got {k:?}"
            );
        } else if i == 9 {
            assert!(
                matches!(
                    k,
                    super::Gemma4LayerKind::GlobalPaged {
                        group_id: 0,
                        paged_idx: 1
                    }
                ),
                "layer 9 must be GlobalPaged{{1}}, got {k:?}"
            );
        } else {
            assert!(
                matches!(k, super::Gemma4LayerKind::SlidingPaged { group_id: 1, .. }),
                "layer {i} must be SlidingPaged in group 1, got {k:?}"
            );
        }
    }
}

#[cfg(test)]
pub(crate) fn cast_paged_tiny_weights_to_bf16(inner: &mut super::Gemma4Inner) {
    use crate::array::{DType, MxArray};
    let cast =
        |array: &MxArray| -> MxArray { array.astype(DType::BFloat16).expect("astype BFloat16") };
    let weight = inner.embed_tokens.get_weight();
    inner
        .embed_tokens
        .set_weight(&cast(&weight))
        .expect("embed");
    let weight = inner.final_norm.get_weight();
    inner
        .final_norm
        .set_weight(&cast(&weight))
        .expect("final_norm");
    if let Some(ref mut head) = inner.lm_head {
        let weight = head.get_weight();
        head.set_weight(&cast(&weight), "lm_head").expect("lm_head");
    }
    for layer in &mut inner.layers {
        layer
            .set_input_layernorm_weight(&cast(&layer.input_layernorm_weight()))
            .expect("input norm");
        layer
            .set_post_attention_layernorm_weight(&cast(&layer.post_attention_layernorm_weight()))
            .expect("post attention norm");
        layer
            .set_pre_feedforward_layernorm_weight(&cast(&layer.pre_feedforward_layernorm_weight()))
            .expect("pre ffn norm");
        layer
            .set_post_feedforward_layernorm_weight(&cast(
                &layer.post_feedforward_layernorm_weight(),
            ))
            .expect("post ffn norm");
        let attention = &mut layer.self_attn;
        let weight = attention.q_proj_weight();
        attention.set_q_proj_weight(&cast(&weight)).expect("q");
        let weight = attention.k_proj_weight();
        attention.set_k_proj_weight(&cast(&weight)).expect("k");
        if let Some(weight) = attention.v_proj_weight_opt() {
            attention.set_v_proj_weight(&cast(&weight)).expect("v");
        }
        let weight = attention.o_proj_weight();
        attention.set_o_proj_weight(&cast(&weight)).expect("o");
        let weight = attention.q_norm_weight();
        attention.set_q_norm_weight(&cast(&weight)).expect("qn");
        let weight = attention.k_norm_weight();
        attention.set_k_norm_weight(&cast(&weight)).expect("kn");
        if let crate::models::gemma4::quantized_linear::Gemma4MLPVariant::Standard(ref mut mlp) =
            layer.mlp
        {
            let weight = mlp.gate_proj_weight();
            mlp.set_gate_proj_weight(&cast(&weight)).expect("gate");
            let weight = mlp.up_proj_weight();
            mlp.set_up_proj_weight(&cast(&weight)).expect("up");
            let weight = mlp.down_proj_weight();
            mlp.set_down_proj_weight(&cast(&weight)).expect("down");
        }
    }
}

/// Smoke test for `paged_turn_sync_core` via direct helper drives.
///
/// Random-init weights cast to BF16 (the paged pool's expected
/// dtype). Validates the adapter lifecycle (reset →
/// find_cached_prefix → allocate_suffix → record_tokens →
/// forward_paged_or_flat) and that produced logits have the
/// expected shape, without asserting numerical equivalence to the
/// flat path (random weights). Gracefully skipped on no-Metal.
#[test]
fn test_run_paged_prefill_decode_smoke() {
    // Block-paged needs the Metal backend; on a non-Metal build the
    // adapter is gated off (None) and there is nothing to exercise.
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping (paged backend unavailable without Metal)");
        return;
    }
    let cfg = paged_tiny_config(Some(true));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };
    assert!(inner.kv_cache_coordinator.is_some());
    if let Err(e) = inner.init_caches_sync() {
        eprintln!("init_caches_sync skipped: {}", e.reason);
        return;
    }

    cast_paged_tiny_weights_to_bf16(&mut inner);

    // Coordinated full+sliding lifecycle.
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    if let Some(coordinator) = inner.kv_cache_coordinator.as_mut()
        && let Err(error) = coordinator.reset_scheduled_request(0)
    {
        eprintln!("skipping (coordinator reset failed): {error}");
        return;
    }

    let last_logits = match inner.run_paged_prefill_chunk(&prompt, &prompt, 0, 0, 0, None) {
        Ok(l) => l,
        Err(e) => {
            let msg = e.reason.to_string();
            if msg.contains("No Metal device found") || msg.contains("not supported") {
                eprintln!("skipping smoke: {msg}");
                return;
            }
            panic!("run_paged_prefill_chunk failed: {msg}");
        }
    };
    let vocab = last_logits.shape_at(0).expect("shape");
    assert_eq!(vocab, 100, "vocab_size from paged_tiny_config");

    let mut next_token: u32 = 5;
    for _ in 0..4 {
        match inner.run_paged_decode_step(next_token) {
            Ok(logits) => {
                assert_eq!(logits.shape_at(0).expect("shape"), 1);
                assert_eq!(logits.shape_at(1).expect("shape"), 1);
                assert_eq!(logits.shape_at(2).expect("shape"), 100);
            }
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping decode (no Metal): {msg}");
                    return;
                }
                panic!("run_paged_decode_step failed: {msg}");
            }
        }
        next_token = next_token.wrapping_add(1);
    }

    if let Some(coordinator) = inner.kv_cache_coordinator.as_mut() {
        let _ = coordinator.release_request_all(0);
    }
}

fn prepare_tiny_paged_request(
    inner: &mut super::Gemma4Inner,
    seq_id: u32,
    prompt: &[u32],
) -> Result<()> {
    let coordinator = inner
        .kv_cache_coordinator
        .as_mut()
        .ok_or_else(|| Error::from_reason("tiny Gemma4 paged adapter missing"))?;
    coordinator
        .reset_scheduled_request(seq_id)
        .map_err(Error::from_reason)?;
    inner.run_scheduled_paged_prefill_slice(seq_id, prompt, 0, true)?;
    Ok(())
}

fn tiny_exclusive_next_token(
    inner: &mut super::Gemma4Inner,
    seq_id: u32,
    prompt: &[u32],
    decode_token: u32,
) -> Result<(u32, std::time::Duration)> {
    prepare_tiny_paged_request(inner, seq_id, prompt)?;
    let decode_started = std::time::Instant::now();
    let logits = inner.run_paged_decode_step_for(seq_id, decode_token)?;
    let next = logits.argmax(-1, Some(false))?.item_at_int32(0)? as u32;
    let decode_elapsed = decode_started.elapsed();
    inner
        .kv_cache_coordinator
        .as_mut()
        .expect("adapter prepared")
        .release_request_all(seq_id)
        .map_err(Error::from_reason)?;
    Ok((next, decode_elapsed))
}

/// A real hybrid Gemma wave must execute both rows in one model forward
/// while preserving the greedy result of two isolated serial replays.
/// This is the regression oracle that the former rotating-cache design
/// could not satisfy.
#[test]
fn gemma4_hybrid_n2_batched_decode_matches_serial_rows() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping (paged backend unavailable without Metal)");
        return;
    }
    let config = super::Gemma4Config {
        num_hidden_layers: 4,
        layer_types: vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ],
        num_kv_shared_layers: Some(2),
        ..paged_tiny_config(Some(true))
    };
    let mut inner = match super::Gemma4Inner::new(config) {
        Ok(inner) => inner,
        Err(error) if error.reason.contains("No Metal device found") => {
            eprintln!("skipping (no Metal device): {}", error.reason);
            return;
        }
        Err(error) => panic!("unexpected Gemma4Inner::new failure: {}", error.reason),
    };
    cast_paged_tiny_weights_to_bf16(&mut inner);
    let prompt_a = [3, 5, 7, 9];
    let prompt_b = [4, 6, 8, 10, 12, 14];
    let (baseline_a, exclusive_a) =
        tiny_exclusive_next_token(&mut inner, 1, &prompt_a, 21).expect("exclusive A");
    let (baseline_b, exclusive_b) =
        tiny_exclusive_next_token(&mut inner, 2, &prompt_b, 22).expect("exclusive B");
    prepare_tiny_paged_request(&mut inner, 101, &prompt_a).expect("prefill A");
    prepare_tiny_paged_request(&mut inner, 202, &prompt_b).expect("prefill B");
    let batched_started = std::time::Instant::now();
    let logits = inner
        .run_paged_decode_step_batched(&[(101, 21), (202, 22)])
        .expect("fused N=2 decode");
    logits.eval();
    let batched_elapsed = batched_started.elapsed();
    let batched = logits
        .argmax(-1, Some(false))
        .expect("argmax")
        .to_int32()
        .expect("batched tokens")
        .as_ref()
        .to_vec();
    assert_eq!(batched, vec![baseline_a as i32, baseline_b as i32]);
    eprintln!(
        "gemma4 N=2 hybrid decode: fused={:.3}ms serial={:.3}ms",
        batched_elapsed.as_secs_f64() * 1_000.0,
        (exclusive_a + exclusive_b).as_secs_f64() * 1_000.0,
    );
    let coordinator = inner.kv_cache_coordinator.as_mut().expect("coordinator");
    coordinator
        .eval_pending_pool_writes_all()
        .expect("materialize batched K/V");
    for seq_id in [101, 202] {
        coordinator
            .prune_sliding_all(seq_id)
            .expect("prune sliding group");
        coordinator
            .release_request_all(seq_id)
            .expect("release hybrid request");
    }
}

#[test]
fn test_gemma4_decode_checkpoint_retains_recent_retokenization_drift() {
    let cfg = paged_tiny_config(Some(true));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    let block_size = 16;
    let target_tokens: Vec<u32> = (1000..1016).collect();
    let target_hash = super::compute_gemma4_paged_prefix_block_hash(
        &target_tokens,
        target_tokens.len() as u32,
        block_size,
        0,
    )
    .expect("target hash");
    inner
        .sliding_prefix_checkpoints
        .push_back(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: target_tokens.len() as u32,
            block_size,
            final_block_hash: target_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: target_tokens.clone(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

    // The observed Gemma4 tool-call retokenization drift needed the
    // checkpoint five block boundaries behind the final decode state:
    // 46272 was requested after 46288, 46304, 46320, and 46336 had
    // also been checkpointed.
    for i in 0..4 {
        let tokens: Vec<u32> = (0..16).map(|token| 2000 + i as u32 + token).collect();
        let hash = super::compute_gemma4_paged_prefix_block_hash(
            &tokens,
            tokens.len() as u32,
            block_size,
            0,
        )
        .expect("newer hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: tokens.len() as u32,
                block_size,
                final_block_hash: hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens,
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });
        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
            inner.sliding_prefix_checkpoints.pop_front();
        }
    }

    let restored = inner
        .find_gemma4_sliding_prefix_checkpoint(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("prefix lookup");
    assert!(
        restored.is_some(),
        "decode checkpoints must retain the block needed after modest retokenization drift"
    );
}

#[test]
fn test_gemma4_decode_checkpoint_retains_sliding_window_drift() {
    let mut cfg = paged_tiny_config(Some(true));
    cfg.sliding_window = 512;
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    let block_size = 16;
    let checkpoint_limit =
        super::gemma4_sliding_prefix_checkpoint_limit_for_override(&inner.config, block_size, None);
    assert_eq!(
        checkpoint_limit, 64,
        "512-token sliding window with 16-token blocks should retain two windows of decode checkpoints"
    );
    let target_tokens: Vec<u32> = (3000..3016).collect();
    let target_hash = super::compute_gemma4_paged_prefix_block_hash(
        &target_tokens,
        target_tokens.len() as u32,
        block_size,
        0,
    )
    .expect("target hash");
    inner
        .sliding_prefix_checkpoints
        .push_back(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: target_tokens.len() as u32,
            block_size,
            final_block_hash: target_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: target_tokens.clone(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

    // The live 2026-05-09 Gemma4 trace needed a checkpoint eighteen
    // block boundaries behind the final decode state (57072 requested
    // after decode reached 57360). A one-window default retains that
    // level of retokenization drift instead of forcing a full replay.
    for i in 0..18 {
        let token_base = 4000 + (i as u32 * block_size);
        let tokens: Vec<u32> = (0..block_size).map(|token| token_base + token).collect();
        let hash = super::compute_gemma4_paged_prefix_block_hash(
            &tokens,
            tokens.len() as u32,
            block_size,
            0,
        )
        .expect("newer hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: tokens.len() as u32,
                block_size,
                final_block_hash: hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens,
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });
        while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
            inner.sliding_prefix_checkpoints.pop_front();
        }
    }

    let restored = inner
        .find_gemma4_sliding_prefix_checkpoint(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("prefix lookup");
    assert!(
        restored.is_some(),
        "decode checkpoints must retain one sliding-window worth of retokenization drift"
    );
}

#[test]
fn test_gemma4_decode_checkpoint_retains_auxiliary_branch_interleaving() {
    let mut cfg = paged_tiny_config(Some(true));
    cfg.sliding_window = 1024;
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    let block_size = 16;
    let checkpoint_limit =
        super::gemma4_sliding_prefix_checkpoint_limit_for_override(&inner.config, block_size, None);
    assert_eq!(
        checkpoint_limit, 128,
        "1024-token sliding window with 16-token blocks should retain two windows"
    );
    let target_tokens: Vec<u32> = (10_000..10_016).collect();
    let target_hash = super::compute_gemma4_paged_prefix_block_hash(
        &target_tokens,
        target_tokens.len() as u32,
        block_size,
        0,
    )
    .expect("target hash");
    inner
        .sliding_prefix_checkpoints
        .push_back(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: target_tokens.len() as u32,
            block_size,
            final_block_hash: target_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: target_tokens.clone(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

    // The 2026-05-09 live trace stored the needed 48,416-token
    // checkpoint, then 93 checkpoints from auxiliary 29k/33k branches
    // before the main branch asked for 48,416 again. A one-window FIFO
    // cap evicted it; two windows retains it without unbounded growth.
    for i in 0..93 {
        let token_base = 20_000 + (i as u32 * block_size);
        let tokens: Vec<u32> = (0..block_size).map(|token| token_base + token).collect();
        let hash = super::compute_gemma4_paged_prefix_block_hash(
            &tokens,
            tokens.len() as u32,
            block_size,
            0,
        )
        .expect("newer hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: tokens.len() as u32,
                block_size,
                final_block_hash: hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens,
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });
        super::trim_gemma4_sliding_prefix_checkpoints(
            &mut inner.sliding_prefix_checkpoints,
            super::Gemma4SlidingRetentionCaps::pre_ladder(
                checkpoint_limit,
                super::Gemma4SlidingCheckpointBytes::for_config(&inner.config),
            ),
            false,
        );
    }

    let restored = inner
        .find_gemma4_sliding_prefix_checkpoint(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("prefix lookup");
    assert!(
        restored.is_some(),
        "decode checkpoints must survive auxiliary branch interleaving seen in live sessions"
    );
}

#[test]
fn test_gemma4_sliding_decode_checkpoint_interval_uses_window_stride() {
    let mut cfg = paged_tiny_config(Some(true));
    cfg.sliding_window = 1024;
    assert_eq!(
        super::gemma4_sliding_decode_checkpoint_interval(&cfg, 16),
        1024
    );

    cfg.sliding_window = 1000;
    assert_eq!(
        super::gemma4_sliding_decode_checkpoint_interval(&cfg, 16),
        1008,
        "checkpoint interval should stay aligned to paged block boundaries"
    );
}

#[test]
fn test_gemma4_sliding_prefix_checkpoint_restores_nearest_prefix() {
    let cfg = paged_tiny_config(Some(true));
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    let block_size = 16;
    let tokens: Vec<u32> = (0..1280).map(|token| 50_000 + token).collect();
    let checkpoint_len = 1024;
    let checkpoint_hash =
        super::compute_gemma4_paged_prefix_block_hash(&tokens, checkpoint_len, block_size, 0)
            .expect("checkpoint hash");
    inner
        .sliding_prefix_checkpoints
        .push_back(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: checkpoint_len,
            block_size,
            final_block_hash: checkpoint_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: tokens[..checkpoint_len as usize].to_vec(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

    let hit = inner
        .find_gemma4_sliding_prefix_checkpoint(&tokens, tokens.len() as u32, block_size, 0)
        .expect("prefix lookup")
        .expect("nearest checkpoint hit");
    assert_eq!(hit.prefix_len, checkpoint_len);
}

#[test]
fn test_gemma4_mid_prompt_prefix_hit_uses_near_prefill_checkpoint() {
    let mut cfg = paged_tiny_config(Some(true));
    cfg.sliding_window = 1024;
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    let block_size = 16;
    let cached_prefix_len = 24_352;
    let checkpoint_len = 23_552;
    let tokens: Vec<u32> = (0..cached_prefix_len).map(|token| 90_000 + token).collect();
    let checkpoint_hash =
        super::compute_gemma4_paged_prefix_block_hash(&tokens, checkpoint_len, block_size, 0)
            .expect("checkpoint hash");
    inner
        .sliding_prefix_checkpoints
        .push_back(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: checkpoint_len,
            block_size,
            final_block_hash: checkpoint_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: tokens[..checkpoint_len as usize].to_vec(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

    let hit = inner
        .find_gemma4_sliding_prefix_checkpoint(&tokens, cached_prefix_len, block_size, 0)
        .expect("prefix lookup")
        .expect("near checkpoint hit");
    assert_eq!(hit.prefix_len, checkpoint_len);
    assert_eq!(cached_prefix_len - hit.prefix_len, 800);
    assert_eq!(
        super::gemma4_large_sliding_restore_suppression_limit(
            &inner.config,
            block_size,
            cached_prefix_len - hit.prefix_len
        ),
        None,
        "a one-window prefill checkpoint should prevent cold-prefill suppression"
    );
}

#[test]
fn test_gemma4_window_stride_checkpoints_retain_old_branch_prefix() {
    let mut cfg = paged_tiny_config(Some(true));
    cfg.sliding_window = 1024;
    let mut inner = match super::Gemma4Inner::new(cfg) {
        Ok(i) => i,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("No Metal device found") {
                eprintln!("skipping (no Metal device): {msg}");
                return;
            }
            panic!("unexpected Gemma4Inner::new failure: {msg}");
        }
    };

    let block_size = 16;
    let target_len = 36_096;
    let target_tokens: Vec<u32> = (0..target_len).map(|token| 70_000 + token).collect();
    let target_hash =
        super::compute_gemma4_paged_prefix_block_hash(&target_tokens, target_len, block_size, 0)
            .expect("target hash");
    inner
        .sliding_prefix_checkpoints
        .push_back(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: target_len,
            block_size,
            final_block_hash: target_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: target_tokens.clone(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

    let checkpoint_limit =
        super::gemma4_sliding_prefix_checkpoint_limit_for_override(&inner.config, block_size, None);
    let interval = super::gemma4_sliding_decode_checkpoint_interval(&inner.config, block_size);
    assert_eq!(interval, 1024);
    assert_eq!(checkpoint_limit, 128);

    for i in 0..96 {
        let prefix_len = 80_000 + i as u32 * interval;
        let tokens: Vec<u32> = (0..prefix_len).map(|token| 200_000 + token).collect();
        let hash =
            super::compute_gemma4_paged_prefix_block_hash(&tokens, prefix_len, block_size, 0)
                .expect("newer hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len,
                block_size,
                final_block_hash: hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens,
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });
        super::trim_gemma4_sliding_prefix_checkpoints(
            &mut inner.sliding_prefix_checkpoints,
            super::Gemma4SlidingRetentionCaps::pre_ladder(
                checkpoint_limit,
                super::Gemma4SlidingCheckpointBytes::for_config(&inner.config),
            ),
            false,
        );
    }

    let hit = inner
        .find_gemma4_sliding_prefix_checkpoint(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("prefix lookup")
        .expect("old branch checkpoint hit");
    assert_eq!(hit.prefix_len, target_len);
}

/// KV-shared layers must resolve their anchor's physical ordinal within
/// the correct full/sliding group.
#[test]
fn test_compute_layer_kinds_kv_sharing_resolves_anchors() {
    // 8 layers: pattern S G S G S G S G (4 global @ 1, 3, 5, 7).
    // num_kv_shared_layers = 4 → last 4 (indices 4, 5, 6, 7) reuse anchors.
    // Anchor for shared global at i=5 should be the last non-shared
    // global before first_kv_shared_layer (=4): that's i=3 → paged_idx=1.
    // Anchor for shared sliding at i=4 should be sliding layer i=2,
    // whose physical ordinal within the sliding group is 1.
    let layer_types: Vec<String> = (0..8)
        .map(|i| {
            if i % 2 == 1 {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();
    let cfg = super::Gemma4Config {
        num_hidden_layers: 8,
        layer_types,
        num_kv_shared_layers: Some(4),
        ..paged_tiny_config(None)
    };
    let kinds = super::compute_layer_kinds(&cfg);
    // Non-shared layers: sliding group 1 has ordinals 0,1; full group 0
    // has ordinals 0,1.
    assert!(matches!(
        kinds[0],
        super::Gemma4LayerKind::SlidingPaged {
            group_id: 1,
            paged_idx: 0
        }
    ));
    assert!(matches!(
        kinds[1],
        super::Gemma4LayerKind::GlobalPaged {
            group_id: 0,
            paged_idx: 0
        }
    ));
    assert!(matches!(
        kinds[2],
        super::Gemma4LayerKind::SlidingPaged {
            group_id: 1,
            paged_idx: 1
        }
    ));
    assert!(matches!(
        kinds[3],
        super::Gemma4LayerKind::GlobalPaged {
            group_id: 0,
            paged_idx: 1
        }
    ));
    // Shared layers 4..8 are aliases. They do not consume paged slots;
    // SharedOnGlobal carries the ANCHOR's pool slot, and
    // SharedOnSliding carries the anchor's group-local physical ordinal.
    match kinds[4] {
        super::Gemma4LayerKind::SharedOnSliding {
            group_id,
            anchor_paged_idx,
        } => {
            assert_eq!(group_id, 1);
            assert_eq!(anchor_paged_idx, 1, "anchor for sliding-shared layer 4");
        }
        ref other => panic!("layer 4: expected SharedOnSliding, got {other:?}"),
    }
    match kinds[5] {
        super::Gemma4LayerKind::SharedOnGlobal {
            group_id,
            anchor_paged_idx,
        } => {
            assert_eq!(group_id, 0);
            // Anchor at layer 3 → paged_idx 1.
            assert_eq!(anchor_paged_idx, 1, "anchor paged_idx for global-shared 5");
        }
        ref other => panic!("layer 5: expected SharedOnGlobal, got {other:?}"),
    }
    match kinds[6] {
        super::Gemma4LayerKind::SharedOnSliding {
            group_id,
            anchor_paged_idx,
        } => {
            assert_eq!(group_id, 1);
            assert_eq!(anchor_paged_idx, 1, "anchor for sliding-shared layer 6");
        }
        ref other => panic!("layer 6: expected SharedOnSliding, got {other:?}"),
    }
    match kinds[7] {
        super::Gemma4LayerKind::SharedOnGlobal {
            group_id,
            anchor_paged_idx,
        } => {
            assert_eq!(group_id, 0);
            assert_eq!(anchor_paged_idx, 1, "anchor paged_idx for global-shared 7");
        }
        ref other => panic!("layer 7: expected SharedOnGlobal, got {other:?}"),
    }
}

/// Keep cached and freshly-derived routes identical, including group IDs
/// and group-local physical ordinals.
fn layer_kind_matches(a: &super::Gemma4LayerKind, b: &super::Gemma4LayerKind) -> bool {
    a == b
}

/// `Gemma4Inner::new` must cache `layer_kinds` once instead of
/// re-deriving it (BTreeMap/BTreeSet grouping + a sort, see
/// `compute_layer_kinds_from_kv_cache_specs`) on every paged
/// prefill-chunk / decode-step call. The cached field must always equal
/// a fresh from-scratch computation over the same config — covers
/// all-global, hybrid sliding+global, and KV-shared layouts (mirrors
/// the three `test_compute_layer_kinds_*` cases above).
#[test]
fn test_gemma4_inner_caches_layer_kinds_matching_fresh_compute() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping (paged backend unavailable without Metal)");
        return;
    }

    let all_global = super::Gemma4Config {
        num_hidden_layers: 4,
        layer_types: vec!["full_attention".to_string(); 4],
        ..paged_tiny_config(Some(true))
    };

    let cycle = ["sliding_attention"; 4]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once("full_attention".to_string()))
        .collect::<Vec<_>>();
    let hybrid = super::Gemma4Config {
        num_hidden_layers: 10,
        layer_types: (0..10).map(|i| cycle[i % 5].clone()).collect(),
        ..paged_tiny_config(Some(true))
    };

    let shared_layer_types: Vec<String> = (0..8)
        .map(|i| {
            if i % 2 == 1 {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();
    let kv_shared = super::Gemma4Config {
        num_hidden_layers: 8,
        layer_types: shared_layer_types,
        num_kv_shared_layers: Some(4),
        ..paged_tiny_config(Some(true))
    };

    for cfg in [all_global, hybrid, kv_shared] {
        let expected = super::compute_layer_kinds_from_kv_cache_specs(&cfg)
            .expect("fresh layer-kind computation must succeed for a valid paged config");
        let inner = match super::Gemma4Inner::new(cfg) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };
        assert!(
            inner.kv_cache_coordinator.is_some(),
            "test configs force use_block_paged_cache=true"
        );
        assert_eq!(
            inner.layer_kinds.len(),
            expected.len(),
            "cached layer_kinds length must match a fresh compute"
        );
        for (i, (got, want)) in inner.layer_kinds.iter().zip(expected.iter()).enumerate() {
            assert!(
                layer_kind_matches(got, want),
                "layer {i}: cached layer_kinds diverged from fresh compute: \
                 got {got:?}, want {want:?}"
            );
        }
    }
}

/// Manual timing probe (not a correctness gate — `#[ignore]`d so it
/// never runs in CI). Measures the per-call cost this task eliminates:
/// re-deriving the routing table from scratch (BTreeMap/BTreeSet + sort)
/// vs. the cached `Vec::clone`. Pure CPU, no GPU/model weights, immune
/// to thermal throttling. Run with:
/// `cargo test -p mlx-core --release --lib -- --ignored --nocapture \
///  bench_layer_kinds_manual`
#[test]
#[ignore]
fn bench_layer_kinds_manual() {
    // Scaled to ~48 layers with a realistic 5:1 sliding:global cycle
    // and KV-sharing, so the BTreeMap/BTreeSet grouping + sort has a
    // realistic amount of work to do.
    let cycle = ["sliding_attention"; 4]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once("full_attention".to_string()))
        .collect::<Vec<_>>();
    let cfg = super::Gemma4Config {
        num_hidden_layers: 48,
        layer_types: (0..48).map(|i| cycle[i % 5].clone()).collect(),
        num_kv_shared_layers: Some(8),
        ..paged_tiny_config(Some(true))
    };

    let n: u32 = 200_000;

    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(
            super::compute_layer_kinds_from_kv_cache_specs(std::hint::black_box(&cfg)).unwrap(),
        );
    }
    eprintln!("recompute: {:?}/call", start.elapsed() / n);

    let cached = super::compute_layer_kinds_from_kv_cache_specs(&cfg).unwrap();
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(std::hint::black_box(&cached).clone());
    }
    eprintln!("cached clone: {:?}/call", start.elapsed() / n);
}

#[test]
fn test_compute_layer_kv_cache_specs_group_full_sliding_and_shared_aliases() {
    let layer_types: Vec<String> = (0..8)
        .map(|i| {
            if i % 2 == 1 {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();
    let cfg = super::Gemma4Config {
        num_hidden_layers: 8,
        layer_types,
        num_kv_shared_layers: Some(4),
        sliding_window: 17,
        max_position_embeddings: 128,
        ..paged_tiny_config(None)
    };

    let specs =
        super::compute_layer_kv_cache_specs(&cfg, 8, super::KVCacheDType::BFloat16).unwrap();
    assert_eq!(specs.len(), 8);
    assert_eq!(specs[4].shared_kv_anchor, Some(2));
    assert_eq!(specs[5].shared_kv_anchor, Some(3));
    assert_eq!(super::physical_full_attention_layer_count(&specs), 2);

    let groups =
        super::compute_layer_kv_cache_groups(&cfg, 8, super::KVCacheDType::BFloat16, 32).unwrap();
    let full_group = groups
        .iter()
        .find(|group| matches!(group.attention_kind, super::AttentionKind::Full))
        .expect("full group");
    assert_eq!(full_group.layer_indices, vec![1, 3, 5, 7]);
    assert_eq!(full_group.physical_layer_indices, vec![1, 3]);

    let sliding_group = groups
        .iter()
        .find(|group| {
            matches!(
                group.attention_kind,
                super::AttentionKind::SlidingWindow { sliding_window: 17 }
            )
        })
        .expect("sliding group");
    assert_eq!(sliding_group.layer_indices, vec![0, 2, 4, 6]);
    assert_eq!(sliding_group.physical_layer_indices, vec![0, 2]);
    assert_eq!(
        sliding_group.max_admission_blocks, 7,
        "ceil((17 - 1 + 32) / 8) + one partial block"
    );
}
