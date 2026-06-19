//! Prefix-cache verification and post-turn cache-state persistence,
//! plus the multimodal (image) cache-key helpers.

use std::hash::{DefaultHasher, Hash, Hasher};

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
mod save_cache_state_after_delta_tests {
    //! Guards the sticky-`cached_image_key` invariant on the text-only
    //! delta path. Calling `save_cache_state_direct(has_images: false,
    //! ...)` after a delta continuation would clear `cached_image_key`
    //! even though the live KV cache still encodes the prior prefill's
    //! image attention state — contradicting the TS `ChatSession` routing
    //! contract (warm cache across text-only follow-ups) and making the
    //! delta path fail with a cryptic "chat_tokens_delta_sync is
    //! text-only; session currently holds image state" on the next turn.
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
