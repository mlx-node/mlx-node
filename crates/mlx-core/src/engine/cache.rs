//! Prefix-cache verification and post-turn cache-state persistence,
//! plus the multimodal (image) cache-key helpers.

use std::hash::{DefaultHasher, Hash, Hasher};

use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;

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

/// Hash raw image bytes to a u64 key for cache lookup.
fn hash_image_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Combine individual image hashes into a single cache key.
/// Order matters: different orderings of the same images produce different keys.
fn combine_image_hashes(hashes: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for h in hashes {
        h.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute a combined cache key from raw image bytes.
pub(crate) fn compute_image_cache_key(all_images: &[Vec<u8>]) -> u64 {
    let individual_hashes: Vec<u64> = all_images.iter().map(|img| hash_image_bytes(img)).collect();
    combine_image_hashes(&individual_hashes)
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
/// produce different hashes). Today's Qwen3.5 paged dispatch is text-only
/// (image-bearing turns are routed to the flat path), so the production
/// call always passes `&[]` here. The hook stays in place so that when
/// VLM-paged forward integration lands, the call site only needs to swap
/// in the real image positions.
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
    crate::transformer::paged_kv_cache_adapter::compute_per_block_image_extra_keys(
        token_image_positions,
        num_blocks,
        block_size,
    )
}

/// Whether the compiled init should re-apply the saved M-RoPE offset
/// (`cached_rope_deltas`) after building the decode graph.
///
/// The offset is saved only when a VLM prefill ran, so `has_saved_delta`
/// is effectively "the live KV cache encodes image attention". Two
/// callers need to re-apply it:
///   - **Fresh VLM prefill reusing a cached prefix** (`has_images &&
///     cached_prefix_len > 0`): the new turn shares its image grid with
///     the cached one, and the saved offset carries the image-adjusted
///     M-RoPE position forward into the rebuilt compiled graph.
///   - **Session delta continuation** (`is_delta`): the delta prefill
///     just ran on top of the live KV caches, which still encode the
///     prior VLM prefill's image attention. Without re-applying the
///     offset, the newly-built compiled graph would decode at a
///     sequential M-RoPE position and misposition all generated tokens
///     relative to the cached image patches.
///
/// Pure function — extracted so the decision can be unit-tested
/// without instantiating the compiled decoder.
pub(crate) fn should_reapply_rope_delta(
    has_saved_delta: bool,
    is_delta: bool,
    has_images: bool,
    cached_prefix_len: usize,
) -> bool {
    has_saved_delta && (is_delta || (has_images && cached_prefix_len > 0))
}

/// Whether the compiled init should clear `cached_rope_deltas` after
/// building the decode graph.
///
/// Only fresh text-only prefills clear the offset: they signal that the
/// non-delta cache-prefix verify dropped any prior image-bearing cache,
/// so the stored offset is stale. Delta continuations preserve the
/// offset so chained text-only turns on an image session keep the
/// image-adjusted M-RoPE position.
///
/// Pure function — extracted so the decision can be unit-tested.
pub(crate) fn should_clear_rope_delta(is_delta: bool, has_images: bool) -> bool {
    !has_images && !is_delta
}

/// Direct-ownership version of `save_cache_state` for dedicated-thread models.
///
/// Takes `&mut` refs instead of `Arc<RwLock<>>`. Used by Qwen3.5 Dense on
/// its dedicated model thread.
pub(crate) fn save_cache_state_direct(
    reuse_cache: bool,
    has_images: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    tokens: &[u32],
    expanded_tokens: Option<&[u32]>,
    image_cache_key: u64,
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) {
    if reuse_cache {
        let mut full_history = if has_images {
            expanded_tokens.unwrap_or(tokens).to_vec()
        } else {
            tokens.to_vec()
        };
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
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

/// Commit session state after a text-only delta continuation.
///
/// The delta path (`chat_tokens_delta_sync` / `chat_stream_tokens_delta_sync`)
/// appends a text delta on top of the live KV caches without touching the
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_cache_state_after_delta(
    reuse_cache: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    save_tokens: &[u32],
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) {
    if reuse_cache {
        let mut full_history = save_tokens.to_vec();
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
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
/// Neither has been done — the invariant here is the sole reason the
/// refactor that moves `reset_caches_sync()` from the outer session-start
/// path into the `cached_prefix_len == 0` branch of `chat_sync_core` is
/// safe for Qwen3.5 Dense and MoE.
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
mod save_cache_state_after_delta_tests {
    //! Guards the sticky-`cached_image_key` invariant on the text-only
    //! delta path. Before the fix, `save_cache_state_direct(has_images:
    //! false, ...)` was called after every delta continuation, which
    //! cleared `cached_image_key` even though the live KV cache still
    //! encoded the prior prefill's image attention state. That
    //! contradicted the TS `ChatSession` routing contract (warm cache
    //! across text-only follow-ups) and caused the delta path to fail
    //! with a cryptic "chat_tokens_delta_sync is text-only; session
    //! currently holds image state" on the very next turn.
    use super::save_cache_state_after_delta;

    #[test]
    fn delta_preserves_cached_image_key_on_reuse_cache_true() {
        let mut cached_history: Vec<u32> = vec![1, 2, 3];
        let mut cached_image_key: Option<u64> = Some(0xdeadbeef);
        let mut cached_rope_deltas: Option<i32> = Some(5);
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> =
            Some(vec![super::Qwen3_5LayerCache::new_full_attention()]);

        save_cache_state_after_delta(
            /* reuse_cache */ true,
            /* generated_tokens */ &[10, 11],
            /* finish_reason */ "stop",
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
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> = None;

        save_cache_state_after_delta(
            true,
            &[10, 11, 12],
            "length",
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
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> =
            Some(vec![super::Qwen3_5LayerCache::new_linear()]);

        save_cache_state_after_delta(
            false,
            &[10],
            "stop",
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
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> = None;

        save_cache_state_after_delta(
            true,
            &[42],
            "stop",
            &[1, 2],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert_eq!(cached_image_key, None);
        assert_eq!(cached_history, vec![1, 2, 42]);
    }
}

#[cfg(test)]
mod rope_delta_gate_tests {
    //! Guards the M-RoPE offset lifecycle across the compiled decode
    //! init branch. The prior bug hard-coded `has_images: false` on the
    //! delta path and unconditionally cleared `cached_rope_deltas`,
    //! which caused the compiled graph to decode text-only deltas at a
    //! sequential position instead of the image-adjusted position —
    //! mispositioning every generated token relative to the cached
    //! image patches baked in by the earlier VLM prefill.
    use super::{should_clear_rope_delta, should_reapply_rope_delta};

    // ---- should_reapply_rope_delta ----

    #[test]
    fn reapply_skipped_when_no_saved_delta() {
        // Text-only session, nothing to re-apply.
        assert!(!should_reapply_rope_delta(false, false, false, 0));
        // Image session with delta, but saved offset missing (fresh VLM
        // prefill clears it before setting — we never enter the gated
        // branch without a saved offset).
        assert!(!should_reapply_rope_delta(false, true, false, 0));
        assert!(!should_reapply_rope_delta(false, false, true, 100));
    }

    #[test]
    fn reapply_fires_on_fresh_vlm_cache_prefix_reuse() {
        // Fresh VLM prefill reusing a cached prefix: both `has_images`
        // AND a non-zero `cached_prefix_len` must be present. The saved
        // offset was written on the prior turn's VLM prefill, so a
        // matching key + prefix means we rebuild the compiled graph at
        // the same image-adjusted position.
        assert!(should_reapply_rope_delta(true, false, true, 100));
    }

    #[test]
    fn reapply_skipped_on_fresh_vlm_without_prefix_match() {
        // VLM prefill without prefix reuse (cached_prefix_len == 0):
        // the compiled init already ran the fresh prefill path, which
        // computed the offset from scratch via M-RoPE. No re-apply.
        assert!(!should_reapply_rope_delta(true, false, true, 0));
    }

    #[test]
    fn reapply_skipped_on_fresh_text_prefill() {
        // Fresh text prefill with no image state: the cache-prefix
        // verify already dropped any prior image-bearing cache, so the
        // saved offset is stale. `should_clear_rope_delta` handles that
        // case by nulling it; re-apply stays off.
        assert!(!should_reapply_rope_delta(true, false, false, 50));
        assert!(!should_reapply_rope_delta(true, false, false, 0));
    }

    #[test]
    fn reapply_fires_on_delta_continuation_with_saved_offset() {
        // THE invariant this fix introduces: delta continuations on an
        // image-bearing session re-apply the saved offset regardless of
        // `has_images` (which is always false on the delta path by
        // construction — delta prefills are text-only) and regardless
        // of `cached_prefix_len` (which is always 0 on the delta path
        // because the live KV cache already contains the full prior
        // history and the delta bypasses the prefix-match flow).
        assert!(should_reapply_rope_delta(true, true, false, 0));
    }

    #[test]
    fn reapply_fires_on_chained_delta_turns() {
        // Chained text-only deltas on the same image session: each
        // turn's compiled init must re-apply the offset so the session
        // stays positioned correctly. The save helper preserves
        // `cached_rope_deltas` on the reuse_cache branch, so the next
        // turn sees `has_saved_delta=true`.
        assert!(should_reapply_rope_delta(true, true, false, 0));
    }

    // ---- should_clear_rope_delta ----

    #[test]
    fn clear_fires_only_on_fresh_text_prefill() {
        // The ONE case where the saved offset is stale: a non-delta
        // text prefill. The cache-prefix verify already dropped any
        // prior image cache, so the offset has nothing valid to apply
        // to on the next turn.
        assert!(should_clear_rope_delta(false, false));
    }

    #[test]
    fn clear_skipped_on_delta_path() {
        // Delta continuations (text-only by construction) preserve the
        // offset — regression gate for the bug this fix addresses. The
        // live KV cache still encodes the prior VLM prefill's image
        // attention, so the next delta turn (and the one after that)
        // must re-apply the same saved offset.
        assert!(!should_clear_rope_delta(true, false));
    }

    #[test]
    fn clear_skipped_on_vlm_prefill() {
        // VLM prefill sets a fresh offset and must not nuke it after
        // init. The `is_delta` axis is false on the non-delta prefill
        // path; the `has_images` axis guards the clear.
        assert!(!should_clear_rope_delta(false, true));
    }

    #[test]
    fn clear_skipped_on_vlm_delta_combination() {
        // Belt-and-suspenders: even if a future caller ever set
        // `is_delta=true, has_images=true`, the clear stays off. No
        // current caller does this — the delta path rejects images at
        // entry — but the gate is written defensively.
        assert!(!should_clear_rope_delta(true, true));
    }
}

#[cfg(test)]
mod verify_cache_prefix_invariant_tests {
    //! Guards the all-or-nothing return-value invariant of
    //! `verify_cache_prefix_direct` documented on its rustdoc. The Qwen3.5
    //! chat_session_start refactor — which moves the unconditional
    //! `reset_caches_sync()` out of the outer session-start path and
    //! relies on verify returning either `0` or the full cached length
    //! to drive the in-core reset-on-miss branch — is **only** safe as
    //! long as this function never returns a mid-sequence prefix length.
    //! A regression here would silently let the caller resume decoding on
    //! a GDN recurrent state that no longer corresponds to the token
    //! prefix in the KV cache, corrupting every generated token.
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
