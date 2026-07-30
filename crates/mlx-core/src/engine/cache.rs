//! Prefix-cache verification and post-turn cache-state persistence,
//! plus the multimodal (image) cache-key helpers.

use sha2::{Digest, Sha256};

use crate::transformer::paged_kv_cache_adapter::PagedTurnPlan;

const IMAGE_PAYLOAD_DIGEST_DOMAIN_V1: &[u8] = b"mlx-node:image-payload-digest:v1\0";
const IMAGE_SET_HOT_LINEAGE_DOMAIN_V1: &[u8] = b"mlx-node:image-set-hot-lineage:v1\0";

/// Stable image-bearing `extra_keys` layout marker (`IMGKEY01` in LE bytes).
///
/// Bump this when the word layout emitted by [`build_paged_extra_keys`] changes.
const IMAGE_BLOCK_EXTRA_KEYS_FORMAT_V1: u64 = u64::from_le_bytes(*b"IMGKEY01");

/// Image preprocessing semantics marker (`PREPR001` in LE bytes).
///
/// Bump this whenever raw-payload decoding, resizing, placeholder expansion, or
/// vision-embedding semantics change in a way that can alter cached K/V state.
/// Keeping it at the image-bearing block-key seam invalidates durable entries
/// across such changes without perturbing text-only keys.
const IMAGE_KV_PREPROCESSING_SEMANTICS_V1: u64 = u64::from_le_bytes(*b"PREPR001");

/// Stable, collision-resistant identity of one raw image payload.
///
/// Persistent paged-prefix keys carry all 256 bits. The compact `u64` image-set
/// key returned alongside these digests is only for same-process hot lineage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ImageCacheDigest([u8; 32]);

impl ImageCacheDigest {
    fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn extra_key_words(self) -> [u64; 4] {
        let mut words = [0u64; 4];
        for (word, bytes) in words.iter_mut().zip(self.0.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().expect("SHA-256 word is 8 bytes"));
        }
        words
    }

    #[cfg(test)]
    pub(crate) fn from_test_word(word: u64) -> Self {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&word.to_le_bytes());
        Self(bytes)
    }
}

/// Load-bearing typed error prefix used when `chat_session_continue_sync`
/// rejects an image parameter because images are changing mid-session.
///
/// Wire contract: when the Rust session-continue path detects that the
/// caller is trying to switch the active image set after a session has
/// already been initialized with different images, it returns a
/// `napi::Error` whose message begins with this prefix. The TypeScript
/// session layer pattern-matches the prefix to recognize the condition
/// and trigger an image-change restart (tearing down the old session
/// state and re-entering the `chat_session_start` path).
///
/// Because TS matches on the literal prefix, this constant MUST NOT
/// change without a coordinated update on both sides of the NAPI
/// boundary.
pub(crate) const IMAGE_CHANGE_RESTART_PREFIX: &str = "IMAGE_CHANGE_REQUIRES_SESSION_RESTART:";

/// Candidate-vs-effective prefix decision for a hybrid image prefill.
///
/// Paged K/V and recurrent GDN state form one logical cache entry. A K/V hit is
/// only effective when the matching GDN sidecar was restored; otherwise the
/// caller must restart the adapter cold while retaining the already prepared
/// vision embeddings for a position-zero prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VlmPagedPrefixResolution {
    pub candidate_cached_prefix_len: u32,
    pub effective_plan: PagedTurnPlan,
    pub gdn_prefix_already_primed: bool,
    pub downgraded_to_cold: bool,
}

pub(crate) fn vlm_prefix_requires_cold_restart(
    candidate_cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
) -> bool {
    candidate_cached_prefix_len > 0 && !gdn_prefix_already_primed
}

pub(crate) fn resolve_vlm_paged_prefix(
    candidate_plan: PagedTurnPlan,
    gdn_prefix_already_primed: bool,
    restart_cold: impl FnOnce() -> Result<PagedTurnPlan, String>,
) -> Result<VlmPagedPrefixResolution, String> {
    let downgraded_to_cold = vlm_prefix_requires_cold_restart(
        candidate_plan.cached_prefix_len,
        gdn_prefix_already_primed,
    );
    let effective_plan = if downgraded_to_cold {
        let cold_plan = restart_cold()?;
        if cold_plan.cached_prefix_len != 0 {
            return Err(format!(
                "VLM cold restart retained {} cached token(s)",
                cold_plan.cached_prefix_len
            ));
        }
        cold_plan
    } else {
        candidate_plan
    };

    Ok(VlmPagedPrefixResolution {
        candidate_cached_prefix_len: candidate_plan.cached_prefix_len,
        effective_plan,
        gdn_prefix_already_primed: !downgraded_to_cold && gdn_prefix_already_primed,
        downgraded_to_cold,
    })
}

/// Hash raw image bytes with an explicitly versioned, process-stable digest.
fn hash_image_payload(bytes: &[u8]) -> ImageCacheDigest {
    let mut hasher = Sha256::new();
    hasher.update(IMAGE_PAYLOAD_DIGEST_DOMAIN_V1);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    ImageCacheDigest(hasher.finalize().into())
}

fn hash_image_payloads<T: AsRef<[u8]>>(all_images: &[T]) -> Vec<ImageCacheDigest> {
    all_images
        .iter()
        .map(|image| hash_image_payload(image.as_ref()))
        .collect()
}

/// Compute a compact hot-lineage key and one stable digest per raw image.
///
/// The returned vector preserves input order and is the identity used by
/// image-aware paged-prefix cache keys. The first tuple field combines that same
/// vector for same-process live-session lineage only. Durable block keys must
/// use the full per-image digests. Each raw image is hashed exactly once.
pub(crate) fn compute_image_cache_keys<T: AsRef<[u8]>>(
    all_images: &[T],
) -> (u64, Vec<ImageCacheDigest>) {
    let per_image_digests = hash_image_payloads(all_images);
    (combine_image_hashes(&per_image_digests), per_image_digests)
}

/// Combine image digests into a compact, deterministic hot-lineage key.
///
/// This key is intentionally not used by persistent block identity.
/// Order matters: different orderings of the same images produce different keys.
fn combine_image_hashes(digests: &[ImageCacheDigest]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(IMAGE_SET_HOT_LINEAGE_DOMAIN_V1);
    hasher.update((digests.len() as u64).to_le_bytes());
    for digest in digests {
        hasher.update(digest.as_bytes());
    }
    let combined: [u8; 32] = hasher.finalize().into();
    u64::from_le_bytes(combined[..8].try_into().expect("SHA-256 prefix is 8 bytes"))
}

/// Compute a combined cache key from raw image bytes.
pub(crate) fn compute_image_cache_key(all_images: &[Vec<u8>]) -> u64 {
    compute_image_cache_keys(all_images).0
}

/// Associate every expanded image-placeholder token with its source digest.
///
/// Qwen-style VLM preprocessing expands each logical image into a known number
/// of placeholder tokens. The counts and digests are both ordered exactly like
/// the raw image list. This helper walks the already-expanded prompt in absolute
/// token order and emits all four digest words at every placeholder position.
/// The flattened `(token_position, digest_word)` representation preserves the
/// existing paged-adapter side-channel type while ensuring persistent block
/// keys receive the complete 256-bit identity.
///
/// The placeholder token id is an argument so this generic cache module does
/// not depend on a Qwen model constant. The mapping rejects ambiguous metadata:
/// there must be one positive token count per image digest, and the total count
/// must exactly equal the number of placeholder ids in `expanded_tokens`.
pub(crate) fn map_expanded_image_token_positions(
    expanded_tokens: &[u32],
    image_token_id: u32,
    per_image_token_counts: &[usize],
    per_image_digests: &[ImageCacheDigest],
) -> Result<Vec<(u32, u64)>, String> {
    if per_image_token_counts.len() != per_image_digests.len() {
        return Err(format!(
            "image metadata cardinality mismatch: {} token counts for {} image digests",
            per_image_token_counts.len(),
            per_image_digests.len(),
        ));
    }

    if let Some((image_index, _)) = per_image_token_counts
        .iter()
        .enumerate()
        .find(|(_, count)| **count == 0)
    {
        return Err(format!(
            "image metadata contains zero expanded tokens for image {image_index}",
        ));
    }

    let expected_placeholders = per_image_token_counts
        .iter()
        .try_fold(0usize, |total, &count| total.checked_add(count))
        .ok_or_else(|| "expanded image placeholder count overflow".to_string())?;
    let actual_placeholders = expanded_tokens
        .iter()
        .filter(|&&token| token == image_token_id)
        .count();
    if actual_placeholders != expected_placeholders {
        return Err(format!(
            "expanded image placeholder mismatch: prompt contains {actual_placeholders}, image metadata requires {expected_placeholders}",
        ));
    }

    let ordered_digests = per_image_token_counts
        .iter()
        .copied()
        .zip(per_image_digests.iter().copied())
        .flat_map(|(count, image_digest)| std::iter::repeat_n(image_digest, count));
    expanded_tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| **token == image_token_id)
        .zip(ordered_digests)
        .map(|((token_position, _), image_digest)| {
            let token_position = u32::try_from(token_position).map_err(|_| {
                format!(
                    "expanded image placeholder position {token_position} exceeds the u32 cache-key range",
                )
            })?;
            Ok((token_position, image_digest))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(|positions| {
            positions
                .into_iter()
                .flat_map(|(token_position, image_digest)| {
                    image_digest
                        .extra_key_words()
                        .into_iter()
                        .map(move |word| (token_position, word))
                })
                .collect()
        })
}

/// Build per-block extra_keys for the paged adapter's prefix-cache walk.
///
/// Multimodal cache isolation: when the prompt contains image tokens,
/// the per-block extra_keys ensure that "same prompt + different image"
/// produces a cache miss (preventing stale-image KV reuse). For
/// text-only prompts (`token_image_positions` is empty), every block gets
/// an empty extra_keys vec — bit-equal to passing `&[]` uniformly to the
/// uniform `find_cached_prefix` / `finalize_turn_keep_live` API.
///
/// `total_tokens` is the FULL prompt length (cached prefix + new suffix
/// the request will write). The number of full blocks covered is
/// `total_tokens / block_size`; the trailing partial block (if any) is
/// not registered until full and so gets no entry here.
///
/// `token_image_positions` should be sorted by `token_pos` for stable
/// hashes (the helper preserves input order; reordered inputs would
/// produce different hashes). [`map_expanded_image_token_positions`] produces
/// this sorted form directly for Qwen-style expanded placeholder runs.
pub(crate) fn build_paged_extra_keys(
    total_tokens: usize,
    block_size: u32,
    token_image_positions: &[(u32, u64)],
) -> Vec<Vec<u64>> {
    let block_size_us = block_size as usize;
    if block_size_us == 0 {
        return Vec::new();
    }
    // Cover every block the request might register (full blocks only).
    // The adapter's per-block API tolerates an over-long vec by indexing
    // only what it needs, so erring high is safe.
    let num_blocks = total_tokens.div_ceil(block_size_us);
    let mut per_block =
        crate::transformer::paged_kv_cache_adapter::compute_per_block_image_extra_keys(
            token_image_positions,
            num_blocks,
            block_size,
        );
    for extra_keys in &mut per_block {
        if !extra_keys.is_empty() {
            extra_keys.insert(0, IMAGE_KV_PREPROCESSING_SEMANTICS_V1);
            extra_keys.insert(0, IMAGE_BLOCK_EXTRA_KEYS_FORMAT_V1);
        }
    }
    per_block
}

/// Direct-ownership version of `save_cache_state` for dedicated-thread models.
///
/// Takes `&mut` refs instead of `Arc<RwLock<>>`. Used by Qwen3.5 Dense on
/// its dedicated model thread.
///
/// `drop_last_always` selects the trailing-token policy, which depends on
/// whether the decode driver forwarded the final committed token into the
/// physical cache:
/// - MTP/macro cores: callers pass `drop_last_always = !last_token_in_cache`.
///   The `decode_loop!` macro and the engine's MTP loop do NOT always forward the
///   stop token before checking it — the AR loop skips the forward on its final
///   step (incl. `max_new_tokens == 1`), and the MTP loop's pre-forward seed
///   re-check and Step-A post-check stop on a token that has not yet been
///   forwarded. Those macros write `last_in_cache = false` on exactly those
///   unforwarded-stop arms, so the helper drops the terminal token when it was
///   unforwarded OR the finish reason is `length`; otherwise it is kept.
/// - `true` (generic `run_decode_loop` flow): the shared loop skips the final
///   committed token's forward on EVERY exit kind, so it is never in the
///   physical cache; the trailing token is dropped unconditionally to keep
///   `cached_token_history.len() == physical_cache_len`. Qwen3.5's GDN
///   recurrent state is non-invertible, so (unlike qwen3/gemma4) it cannot
///   materialize the token with an extra forward and must drop it — same
///   contract as lfm2's `save_cache_state`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_cache_state_direct<C>(
    reuse_cache: bool,
    has_images: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    drop_last_always: bool,
    tokens: &[u32],
    expanded_tokens: Option<&[u32]>,
    image_cache_key: u64,
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<C>>,
) {
    if reuse_cache {
        let mut full_history = if has_images {
            expanded_tokens.unwrap_or(tokens).to_vec()
        } else {
            tokens.to_vec()
        };
        let drop_last = drop_last_always || finish_reason == "length";
        let history_tokens = if drop_last && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        *cached_token_history = full_history;
        *cached_image_key = if has_images {
            Some(image_cache_key)
        } else {
            None
        };
    } else {
        *caches = None;
        cached_token_history.clear();
        *cached_image_key = None;
        *cached_rope_deltas = None;
    }
}

/// Commit session state after a model-specific text-only suffix continuation.
///
/// A specialized model path may append a text suffix on top of live KV caches
/// without touching the
/// image attention state baked in by the preceding prefill. The "current
/// turn is text-only" signal (`has_images == false`) MUST NOT be conflated
/// with "the session has no image context" — the KV caches still encode
/// every image patch from the earlier `chat_session_start` / VLM prefill,
/// and clearing `cached_image_key` here would make the next cache-prefix
/// verify think the session is pure text and accept a future image-carrying
/// turn via the delta path (which produces garbage because the mrope
/// offset `cached_rope_deltas` is stale for the new image grid).
///
/// This helper is identical to [`save_cache_state_direct`] except that it
/// leaves `cached_image_key` untouched on the `reuse_cache=true` branch.
/// The full-reset `reuse_cache=false` branch still clears everything —
/// same invariant as the prefill helper.
///
/// `drop_last_always` has the same meaning as in [`save_cache_state_direct`]:
/// `true` (generic `run_decode_loop` flow) drops the never-forwarded final
/// token on every exit; the MTP/macro cores pass `!last_token_in_cache`, which
/// drops the terminal token when the macro stopped on an unforwarded token
/// (final-step AR stop, MTP seed / Step-A stop) OR the finish reason is
/// `length`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_cache_state_after_delta<C>(
    reuse_cache: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    drop_last_always: bool,
    save_tokens: &[u32],
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<C>>,
) {
    if reuse_cache {
        let mut full_history = save_tokens.to_vec();
        let drop_last = drop_last_always || finish_reason == "length";
        let history_tokens = if drop_last && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        *cached_token_history = full_history;
        // `cached_image_key` intentionally preserved — see doc comment.
    } else {
        *caches = None;
        cached_token_history.clear();
        *cached_image_key = None;
        *cached_rope_deltas = None;
    }
}

/// Direct-ownership version of `verify_cache_prefix` for dedicated-thread models.
///
/// Takes direct refs instead of `Arc<RwLock<>>`. Used by Qwen3.5 Dense on
/// its dedicated model thread.
///
/// # Return-value invariant (load-bearing)
///
/// This helper returns **either `0` (cache miss — caller MUST reset caches
/// before prefill) or `cached.len()` (exact-append hit — the new prompt
/// strictly extends the cached history)**. It **never** returns an
/// intermediate value such as "the first K tokens match, rewind to K".
///
/// That all-or-nothing contract is what makes it safe to drive Qwen3.5's
/// **hybrid linear + attention stack**. The Gated Delta Net (GDN) layers
/// carry a *recurrent* state (`conv_state`, `recurrent_state` in
/// [`crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache::Linear`]) that folds every
/// absorbed token irreversibly into its hidden state — unlike a standard
/// KV cache, a GDN cache **cannot be trimmed or rewound mid-sequence**
/// without corrupting the representation. A non-zero return from this
/// function therefore always means "the incoming tokens are a *pure append*
/// on top of the cached state; continue decoding from the current live
/// caches". No mid-sequence rewind ever happens.
///
/// Any future modification that would relax this contract (e.g. returning
/// a prefix count less than `cached.len()`) MUST simultaneously ensure the
/// caller either (a) restricts the relaxation to pure-KVCache models or
/// (b) introduces GDN-state checkpointing to enable mid-sequence rewinds.
/// Neither has been done — the invariant here is the sole reason it is
/// safe for Qwen3.5 Dense and MoE to run `reset_caches_sync()` only in
/// the `cached_prefix_len == 0` (miss) branch of the session turn core
/// rather than unconditionally on the session-start path.
///
/// ## Sanctioned exception (pure-KV only): qwen3 flat exact-match rewind
///
/// One family-side relaxation under clause (a) is on the books: qwen3's
/// FLAT path (a pure standard-KV stack with an explicit `cache_idx`
/// write pointer) handles the exact-match corner by rewinding one slot
/// and re-forwarding the last token ("Zero delta — re-run last token"
/// in `models/qwen3/model.rs`). Its `ChatBackend::verify_cache_prefix`
/// impl MAY therefore return `cached.len() - 1` on an exact match —
/// see the trait rustdoc in [`crate::engine::backend`] for the exact
/// shape. This helper itself stays all-or-nothing; the exception lives
/// only in that family impl, and is forbidden for any cache with
/// recurrent (GDN / conv) state.
pub(crate) fn verify_cache_prefix_direct(
    reuse_cache: bool,
    has_images: bool,
    tokens: &[u32],
    tokens_for_matching: &[u32],
    image_cache_key: u64,
    cached_token_history: &[u32],
    cached_image_key: &Option<u64>,
    has_caches: bool,
) -> usize {
    if !reuse_cache {
        return 0;
    }
    let cached = cached_token_history;
    if has_images {
        if let Some(cached_key) = *cached_image_key
            && cached_key == image_cache_key
            && !cached.is_empty()
            && tokens_for_matching.len() >= cached.len()
            && tokens_for_matching[..cached.len()] == cached[..]
            && has_caches
        {
            return cached.len();
        }
        0
    } else if !cached.is_empty()
        && tokens.len() >= cached.len()
        && tokens[..cached.len()] == cached[..]
        && has_caches
    {
        cached.len()
    } else {
        0
    }
}

#[cfg(test)]
mod image_cache_identity_tests {
    use super::{
        IMAGE_BLOCK_EXTRA_KEYS_FORMAT_V1, IMAGE_KV_PREPROCESSING_SEMANTICS_V1, ImageCacheDigest,
        build_paged_extra_keys, combine_image_hashes, compute_image_cache_key,
        compute_image_cache_keys, map_expanded_image_token_positions, resolve_vlm_paged_prefix,
        vlm_prefix_requires_cold_restart,
    };
    use crate::transformer::paged_kv_cache_adapter::{PagedTurnPlan, PagedTurnPlanReason};
    use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheKey, ColdGroup};

    const IMAGE_TOKEN_ID: u32 = 248_056;

    fn digest(word: u64) -> ImageCacheDigest {
        ImageCacheDigest::from_test_word(word)
    }

    fn cold_key(extra_keys: &[u64]) -> ColdCacheKey {
        ColdCacheKey::chain(
            ColdGroup::Kv,
            ColdCacheFingerprint::from_components([b"image-cache-test".as_slice()]),
            None,
            &[1, IMAGE_TOKEN_ID, 2, 3],
            extra_keys,
            0,
            0,
        )
    }

    #[test]
    fn image_kv_candidate_requires_cold_restart_without_gdn_sidecar() {
        assert!(!vlm_prefix_requires_cold_restart(0, false));
        assert!(!vlm_prefix_requires_cold_restart(16, true));
        assert!(vlm_prefix_requires_cold_restart(16, false));

        let candidate = PagedTurnPlan {
            cached_prefix_len: 16,
            continued_live_prefix: false,
            allocated_blocks: 1,
            cached_blocks: 1,
            total_budget: 32,
            suffix_len: 16,
            reason: PagedTurnPlanReason::FreshReset,
        };
        let cold = PagedTurnPlan {
            cached_prefix_len: 0,
            continued_live_prefix: false,
            allocated_blocks: 2,
            cached_blocks: 0,
            total_budget: 32,
            suffix_len: 32,
            reason: PagedTurnPlanReason::FreshReset,
        };
        let mut restarted = false;
        let resolution = resolve_vlm_paged_prefix(candidate, false, || {
            restarted = true;
            Ok(cold)
        })
        .unwrap();
        assert!(restarted);
        assert!(resolution.downgraded_to_cold);
        assert_eq!(resolution.candidate_cached_prefix_len, 16);
        assert_eq!(resolution.effective_plan, cold);
        assert!(!resolution.gdn_prefix_already_primed);

        let mut restarted_exact = false;
        let exact = resolve_vlm_paged_prefix(candidate, true, || {
            restarted_exact = true;
            Ok(cold)
        })
        .unwrap();
        assert!(!restarted_exact);
        assert!(!exact.downgraded_to_cold);
        assert_eq!(exact.effective_plan, candidate);
        assert!(exact.gdn_prefix_already_primed);
    }

    #[test]
    fn image_payload_digest_is_version_pinned_and_content_sensitive() {
        let image_a: &[u8] = &[1, 2, 3, 4];
        let changed: &[u8] = &[1, 2, 3, 5];

        let (_, first) = compute_image_cache_keys(&[image_a]);
        let (_, same_bytes_new_slice) = compute_image_cache_keys(&[&[1, 2, 3, 4][..]]);
        let (_, changed_digest) = compute_image_cache_keys(&[changed]);

        assert_eq!(first, same_bytes_new_slice);
        assert_ne!(first, changed_digest);
        assert_eq!(
            first[0].0,
            [
                0x45, 0x42, 0x12, 0xd9, 0x36, 0xe6, 0x52, 0x67, 0xe4, 0xf0, 0x29, 0x71, 0xff, 0x94,
                0x74, 0x9f, 0xf2, 0x7c, 0x13, 0xfb, 0x7c, 0x76, 0x31, 0xa0, 0xbc, 0x33, 0x39, 0x6a,
                0x6a, 0x53, 0xbd, 0x09,
            ],
            "the versioned SHA-256 derivation is part of the durable cache ABI",
        );
    }

    #[test]
    fn image_order_changes_hot_lineage_and_preserves_ordered_digests() {
        let images = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let (combined, individual) = compute_image_cache_keys(&images);
        assert_eq!(combined, combine_image_hashes(&individual));
        assert_eq!(compute_image_cache_key(&images), combined);
        assert_eq!(combined, 0xe6e1_07bd_8bfe_ed56);

        let reversed = vec![images[1].clone(), images[0].clone()];
        let (reversed_combined, reversed_individual) = compute_image_cache_keys(&reversed);
        assert_eq!(reversed_individual, vec![individual[1], individual[0]]);
        assert_ne!(
            combined, reversed_combined,
            "combined image-set key must preserve image order",
        );
    }

    #[test]
    fn expanded_placeholders_carry_every_digest_word_in_prompt_order() {
        let expanded_tokens = [
            1,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            200,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            201,
        ];
        let positions = map_expanded_image_token_positions(
            &expanded_tokens,
            IMAGE_TOKEN_ID,
            &[2, 3],
            &[digest(0xAAAA), digest(0xBBBB)],
        )
        .expect("valid expanded image metadata");

        let expected_positions = [1u32, 2, 4, 5, 6];
        let expected_digests = [
            digest(0xAAAA),
            digest(0xAAAA),
            digest(0xBBBB),
            digest(0xBBBB),
            digest(0xBBBB),
        ];
        let expected = expected_positions
            .into_iter()
            .zip(expected_digests)
            .flat_map(|(position, digest)| {
                digest
                    .extra_key_words()
                    .into_iter()
                    .map(move |word| (position, word))
            })
            .collect::<Vec<_>>();
        assert_eq!(positions, expected);
    }

    #[test]
    fn expanded_placeholder_mapping_rejects_image_metadata_cardinality_mismatch() {
        let error = map_expanded_image_token_positions(
            &[1, IMAGE_TOKEN_ID],
            IMAGE_TOKEN_ID,
            &[1, 1],
            &[digest(0xAAAA)],
        )
        .expect_err("counts and digests must have identical cardinality");

        assert!(error.contains("2 token counts for 1 image digests"));
    }

    #[test]
    fn expanded_placeholder_mapping_rejects_prompt_count_mismatch() {
        let too_few = map_expanded_image_token_positions(
            &[1, IMAGE_TOKEN_ID, 200],
            IMAGE_TOKEN_ID,
            &[2],
            &[digest(0xAAAA)],
        )
        .expect_err("metadata requiring two placeholders must reject one placeholder");
        assert!(too_few.contains("prompt contains 1, image metadata requires 2"));

        let too_many = map_expanded_image_token_positions(
            &[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200],
            IMAGE_TOKEN_ID,
            &[1],
            &[digest(0xAAAA)],
        )
        .expect_err("metadata requiring one placeholder must reject two placeholders");
        assert!(too_many.contains("prompt contains 2, image metadata requires 1"));
    }

    #[test]
    fn expanded_placeholder_mapping_rejects_zero_count_images() {
        let error =
            map_expanded_image_token_positions(&[1, 200], IMAGE_TOKEN_ID, &[0], &[digest(0xAAAA)])
                .expect_err("an image digest without an expanded token span is ambiguous");

        assert!(error.contains("zero expanded tokens for image 0"));
    }

    #[test]
    fn durable_image_keys_bind_full_digest_and_placeholder_position() {
        let (_, digests) = compute_image_cache_keys(&[&[1, 2, 3, 4][..]]);
        let at_one = map_expanded_image_token_positions(
            &[1, IMAGE_TOKEN_ID, 2, 3],
            IMAGE_TOKEN_ID,
            &[1],
            &digests,
        )
        .unwrap();
        let at_two = map_expanded_image_token_positions(
            &[1, 2, IMAGE_TOKEN_ID, 3],
            IMAGE_TOKEN_ID,
            &[1],
            &digests,
        )
        .unwrap();
        let extra_at_one = build_paged_extra_keys(4, 16, &at_one);
        let extra_at_two = build_paged_extra_keys(4, 16, &at_two);

        assert_eq!(
            &extra_at_one[0][..2],
            &[
                IMAGE_BLOCK_EXTRA_KEYS_FORMAT_V1,
                IMAGE_KV_PREPROCESSING_SEMANTICS_V1,
            ],
        );
        assert_eq!(extra_at_one[0].len(), 2 + 4 * 2);
        let carried_digest = extra_at_one[0][2..]
            .chunks_exact(2)
            .flat_map(|pair| pair[0].to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(carried_digest.as_slice(), digests[0].as_bytes());
        assert!(
            extra_at_one[0][2..]
                .chunks_exact(2)
                .all(|pair| pair[1] == 1),
            "every digest word must be bound to the placeholder position",
        );
        assert_ne!(extra_at_one, extra_at_two);
        assert_ne!(cold_key(&extra_at_one[0]), cold_key(&extra_at_two[0]));

        let (_, changed_digests) = compute_image_cache_keys(&[&[1, 2, 3, 5][..]]);
        let changed_positions = map_expanded_image_token_positions(
            &[1, IMAGE_TOKEN_ID, 2, 3],
            IMAGE_TOKEN_ID,
            &[1],
            &changed_digests,
        )
        .unwrap();
        let changed_extra = build_paged_extra_keys(4, 16, &changed_positions);
        assert_ne!(extra_at_one, changed_extra);
        assert_ne!(cold_key(&extra_at_one[0]), cold_key(&changed_extra[0]));
    }

    #[test]
    fn durable_image_keys_preserve_image_order() {
        let (_, ordered_digests) = compute_image_cache_keys(&[&[1, 2, 3][..], &[4, 5, 6][..]]);
        let (_, reversed_digests) = compute_image_cache_keys(&[&[4, 5, 6][..], &[1, 2, 3][..]]);
        let expanded = [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 7, 8];
        let ordered_positions = map_expanded_image_token_positions(
            &expanded,
            IMAGE_TOKEN_ID,
            &[1, 1],
            &ordered_digests,
        )
        .unwrap();
        let reversed_positions = map_expanded_image_token_positions(
            &expanded,
            IMAGE_TOKEN_ID,
            &[1, 1],
            &reversed_digests,
        )
        .unwrap();
        let ordered_extra = build_paged_extra_keys(expanded.len(), 16, &ordered_positions);
        let reversed_extra = build_paged_extra_keys(expanded.len(), 16, &reversed_positions);

        assert_ne!(ordered_extra, reversed_extra);
        assert_ne!(cold_key(&ordered_extra[0]), cold_key(&reversed_extra[0]));
    }

    #[test]
    fn expanded_placeholder_mapping_accepts_empty_text_only_metadata() {
        let positions = map_expanded_image_token_positions(&[1, 200], IMAGE_TOKEN_ID, &[], &[])
            .expect("empty image metadata must preserve the text-only baseline");
        assert!(positions.is_empty());

        let extra_keys = build_paged_extra_keys(16, 16, &positions);
        assert_eq!(extra_keys, vec![Vec::<u64>::new()]);
        assert_eq!(
            cold_key(&extra_keys[0]),
            cold_key(&[]),
            "text-only ColdCacheKey derivation must remain byte-identical",
        );
    }
}

#[cfg(test)]
mod save_cache_state_after_delta_tests {
    //! Guards the sticky-`cached_image_key` invariant on the text-only
    //! specialized suffix path. Calling `save_cache_state_direct(has_images:
    //! false, ...)` after such a continuation would clear `cached_image_key`
    //! even though the live KV cache still encodes the prior prefill's
    //! image attention state — contradicting the TS `ChatSession` routing
    //! contract (warm cache across text-only follow-ups).
    //! [`save_cache_state_after_delta`] preserves the key instead.
    use super::{save_cache_state_after_delta, save_cache_state_direct};

    /// Stand-in cache element. The helper's reuse_cache=false branch only
    /// does `*caches = None;` and never inspects the element, so a
    /// zero-sized dummy reproduces the exact `is_some()`/`is_none()`
    /// behavior these tests assert without binding the engine to a
    /// concrete model cache type.
    #[derive(Clone)]
    struct DummyCache;

    #[test]
    fn delta_preserves_cached_image_key_on_reuse_cache_true() {
        let mut cached_history: Vec<u32> = vec![1, 2, 3];
        let mut cached_image_key: Option<u64> = Some(0xdeadbeef);
        let mut cached_rope_deltas: Option<i32> = Some(5);
        let mut caches: Option<Vec<DummyCache>> = Some(vec![DummyCache]);

        save_cache_state_after_delta(
            /* reuse_cache */ true,
            /* generated_tokens */ &[10, 11],
            /* finish_reason */ "stop",
            /* drop_last_always */ false,
            /* save_tokens */ &[1, 2, 3, 4],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        // Token history extended: pre-decode snapshot + generated tokens
        assert_eq!(cached_history, vec![1, 2, 3, 4, 10, 11]);
        // Image key preserved — THE invariant under test
        assert_eq!(cached_image_key, Some(0xdeadbeef));
        // Other cache state untouched
        assert_eq!(cached_rope_deltas, Some(5));
        assert!(caches.is_some());
    }

    #[test]
    fn delta_drops_trailing_generated_token_on_length_stop() {
        // Matches `save_cache_state_direct` truncation semantics: if the
        // decode terminated at max_new_tokens, the last generated token
        // was cut off mid-stream and must not be persisted.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = Some(42);
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<DummyCache>> = None;

        save_cache_state_after_delta(
            true,
            &[10, 11, 12],
            "length",
            false,
            &[1, 2],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert_eq!(cached_history, vec![1, 2, 10, 11]);
        assert_eq!(cached_image_key, Some(42));
    }

    #[test]
    fn delta_full_reset_clears_everything_when_reuse_cache_false() {
        // `reuse_cache=false` is the cold-path invariant from the prefill
        // helper — when the caller opts out of cache reuse, every piece
        // of session state must be cleared regardless of whether the
        // image key was previously populated.
        let mut cached_history: Vec<u32> = vec![1, 2, 3];
        let mut cached_image_key: Option<u64> = Some(0xabc);
        let mut cached_rope_deltas: Option<i32> = Some(7);
        let mut caches: Option<Vec<DummyCache>> = Some(vec![DummyCache]);

        save_cache_state_after_delta(
            false,
            &[10],
            "stop",
            false,
            &[1],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert!(cached_history.is_empty());
        assert!(cached_image_key.is_none());
        assert!(cached_rope_deltas.is_none());
        assert!(caches.is_none());
    }

    #[test]
    fn delta_with_text_only_session_keeps_key_none() {
        // Sanity: if the session never had images, the delta must not
        // fabricate a key either.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = None;
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<DummyCache>> = None;

        save_cache_state_after_delta(
            true,
            &[42],
            "stop",
            false,
            &[1, 2],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert_eq!(cached_image_key, None);
        assert_eq!(cached_history, vec![1, 2, 42]);
    }

    #[test]
    fn delta_drop_last_always_drops_trailing_token_on_non_length_stop() {
        // Generic-flow contract (qwen3.5 dense/MoE): the shared
        // `run_decode_loop` never forwards the final committed token into
        // the physical cache, regardless of exit kind. With
        // `drop_last_always = true` the trailing generated token is dropped
        // even on a non-`length` finish (EOS / stop / cancel / repetition),
        // keeping `cached_token_history` aligned with the cache length.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = None;
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<DummyCache>> = None;

        save_cache_state_after_delta(
            /* reuse_cache */ true,
            /* generated_tokens */ &[10, 11],
            /* finish_reason */ "stop",
            /* drop_last_always */ true,
            /* save_tokens */ &[1, 2, 3],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        // Trailing token 11 dropped despite the "stop" finish: only the
        // forwarded prefix (prompt + [10]) is persisted.
        assert_eq!(cached_history, vec![1, 2, 3, 10]);
    }

    #[test]
    fn direct_drop_last_always_drops_trailing_token_on_non_length_stop() {
        // Generic-flow FRESH-turn contract (qwen3.5 dense/MoE
        // `save_cache_state` non-delta branch). The shared `run_decode_loop`
        // never forwards the final committed token, so on a non-`length`
        // finish (EOS / stop / cancel / repetition) `drop_last_always = true`
        // must still drop it, keeping `cached_token_history` aligned with the
        // physical cache length. Mirrors the delta-helper test for the
        // fresh-prefill path.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = None;
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<DummyCache>> = None;

        save_cache_state_direct(
            /* reuse_cache */ true,
            /* has_images */ false,
            /* generated_tokens */ &[10, 11],
            /* finish_reason */ "stop",
            /* drop_last_always */ true,
            /* tokens */ &[1, 2, 3],
            /* expanded_tokens */ None,
            /* image_cache_key */ 0,
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        // Token 11 dropped despite "stop"; prompt + forwarded [10] kept.
        assert_eq!(cached_history, vec![1, 2, 3, 10]);
        // Text-only fresh turn must not fabricate an image key.
        assert_eq!(cached_image_key, None);
    }

    #[test]
    fn direct_legacy_false_keeps_terminal_token_on_non_length_stop() {
        // Macro-core contract (drop_last_always = false): the macro cores
        // forward the EOS/stop token BEFORE the stop check, so it IS in the
        // cache and must be KEPT on a non-`length` finish.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = None;
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<DummyCache>> = None;

        save_cache_state_direct(
            true,
            false,
            &[10, 11],
            "stop",
            /* drop_last_always */ false,
            &[1, 2, 3],
            None,
            0,
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        // All generated tokens kept (the macro forwarded the terminal token).
        assert_eq!(cached_history, vec![1, 2, 3, 10, 11]);
    }

    #[test]
    fn direct_unforwarded_stop_drops_trailing_token() {
        // Macro-core contract for an UNFORWARDED stop token: when the macro
        // stops on a token it never forwarded into the physical KV/GDN cache
        // (final-step AR stop, or the MTP seed / Step-A stop arms), the caller
        // passes `drop_last_always = !last_in_cache` with `last_in_cache =
        // false`. The helper must then drop the terminal token even though the
        // finish reason is `stop`, keeping `cached_token_history` aligned with
        // the physical cache length. The generated output the caller returns is
        // unaffected — only the persisted history is trimmed.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = None;
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<DummyCache>> = None;

        let generated = [10, 11];
        let last_in_cache = false;
        save_cache_state_direct(
            true,
            false,
            &generated,
            "stop",
            /* drop_last_always */ !last_in_cache,
            &[1, 2, 3],
            None,
            0,
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        // Unforwarded terminal token 11 dropped from the persisted history.
        assert_eq!(cached_history, vec![1, 2, 3, 10]);
        // The generated output itself is untouched by the save helper.
        assert_eq!(generated, [10, 11]);
    }
}

#[cfg(test)]
mod verify_cache_prefix_invariant_tests {
    //! Guards the all-or-nothing return-value invariant of
    //! `verify_cache_prefix_direct` documented on its rustdoc. The Qwen3.5
    //! chat_session_start path runs `reset_caches_sync()` only in the
    //! in-core reset-on-miss branch and relies on verify returning either
    //! `0` or the full cached length — which is **only** safe as long as
    //! this function never returns a mid-sequence prefix length. A
    //! regression here would silently let the caller resume decoding on a
    //! GDN recurrent state that no longer corresponds to the token prefix
    //! in the KV cache, corrupting every generated token.
    use super::verify_cache_prefix_direct;

    #[test]
    fn returns_zero_when_reuse_cache_disabled() {
        // `reuse_cache = false` short-circuits; everything else is
        // irrelevant. This is the "caller explicitly opted out" path.
        assert_eq!(
            verify_cache_prefix_direct(
                false,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_when_no_caches() {
        // `has_caches = false` means the model has no live KV caches to
        // resume from — a full prefill is required even if the history
        // matches.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                false,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_empty_history() {
        // First session-start turn: nothing cached yet, so we must
        // prefill the whole prompt.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_first_token_mismatch() {
        // Histories diverge at index 0 — no reusable prefix.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[9, 2, 3, 4],
                &[9, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_midsequence_mismatch() {
        // CRITICAL: histories match for 2 tokens then diverge. The
        // function MUST return 0 (full miss), NOT 2 (partial hit).
        // A partial hit would signal the caller to reuse only the first
        // 2 positions of the KV cache — which for the GDN linear layers
        // would require rewinding the recurrent state, which is
        // impossible. The all-or-nothing contract is what keeps this
        // safe.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 7, 4],
                &[1, 2, 7, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_shorter_new_prompt() {
        // New prompt is shorter than the cached history — can't be a
        // forward extension. Rewinding is infeasible (see above), so
        // return 0 and force a fresh prefill.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2],
                &[1, 2],
                0,
                &[1, 2, 3, 4, 5],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_full_length_on_exact_append_hit() {
        // Happy path: the new prompt is `cached + [extra]`. The function
        // returns `cached.len()` so the caller prefills only the delta
        // tail. This is the whole point of the cache-reuse machinery.
        let cached = vec![1u32, 2, 3, 4];
        let new_prompt = vec![1u32, 2, 3, 4, 5, 6];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &new_prompt,
                &new_prompt,
                0,
                &cached,
                &None,
                true,
            ),
            cached.len(),
        );
    }

    #[test]
    fn returns_full_length_on_exact_match() {
        // Edge case: new prompt is byte-identical to cached. Returns
        // `cached.len()` — the caller's zero-delta guard then takes
        // over (see the matching comment in `qwen3_5/model.rs` and
        // `qwen3_5_moe/model.rs`).
        let cached = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(true, false, &cached, &cached, 0, &cached, &None, true,),
            cached.len(),
        );
    }

    #[test]
    fn returns_zero_on_image_key_mismatch() {
        // VLM path: cached image key differs from the current turn's
        // key — the images changed, so the cached KV state no longer
        // represents the new prompt's image attention. Full reset.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                /* new image key */ 999,
                &cached,
                &Some(42),
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_full_length_on_vlm_image_key_match() {
        // VLM happy path: same images, new text tail. Returns the
        // cached prefix length so the caller prefills only the delta.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4, 5];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                42,
                &cached,
                &Some(42),
                true,
            ),
            cached.len(),
        );
    }

    #[test]
    fn returns_zero_on_vlm_missing_image_key() {
        // VLM turn but cached state carries no image key — the cache
        // came from a prior text-only exchange, not a VLM prefill.
        // Safety requires a fresh VLM prefill, not a reuse.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                42,
                &cached,
                &None,
                true,
            ),
            0,
        );
    }

    /// The contract-level invariant: across a broad sweep of inputs the
    /// return value is ALWAYS either `0` or `cached.len()`. Any
    /// intermediate value would corrupt GDN recurrent state on reuse.
    ///
    /// This property-style sweep is belt-and-suspenders on top of the
    /// targeted unit tests above: even if a future refactor changes
    /// branch structure, the invariant holds by construction.
    #[test]
    fn invariant_return_value_is_always_zero_or_cached_len() {
        let cached = vec![10u32, 20, 30, 40, 50];
        // Every prefix-plus-suffix combination and a selection of
        // divergent inputs.
        let candidates: Vec<Vec<u32>> = vec![
            vec![],
            vec![10],
            vec![10, 20],
            vec![10, 20, 30],
            vec![10, 20, 30, 40],
            cached.clone(),
            [cached.clone(), vec![60]].concat(),
            [cached.clone(), vec![60, 70, 80]].concat(),
            vec![99, 20, 30, 40, 50, 60],
            vec![10, 20, 99, 40, 50, 60],
            vec![10, 20, 30, 40, 99, 60],
        ];

        for candidate in &candidates {
            let result = verify_cache_prefix_direct(
                true, false, candidate, candidate, 0, &cached, &None, true,
            );
            assert!(
                result == 0 || result == cached.len(),
                "invariant violated: result={} for candidate={:?} (expected 0 or {})",
                result,
                candidate,
                cached.len(),
            );
        }
    }
}
