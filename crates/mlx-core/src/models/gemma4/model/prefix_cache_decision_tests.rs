//! Pure-logic coverage of the prefix-cache decision tree — no model
//! load required. The verifier `Gemma4Inner::verify_cache_prefix`
//! returns either `0` (miss) or `cached_token_history.len()` (exact
//! prefix relation). The engine session core
//! (`engine::session::chat_turn_core`) then classifies that
//! value plus the incoming prompt length into
//! [`PrefixCacheDecision::StrictExtendHit`] (warm-reuse, skip the
//! cached prefix, prefill only the tail) vs
//! [`PrefixCacheDecision::Miss`] (reset caches + re-init + full
//! prefill).
//!
//! The four cases covered below pin the invariant: exact-match MUST
//! route to `Miss`, not to `StrictExtendHit`. Treating exact-match as a
//! shortcut would corrupt the next warm-hit turn by advancing cache
//! state to `prompt + last_token` while the history write-back only
//! persists `tokens + generated`. This module guarantees the decision logic
//! stays correct in every CI run without a model dependency.

use super::{PrefixCacheDecision, classify_prefix_cache_decision};

#[test]
fn empty_cache_is_miss() {
    // verify_cache_prefix returned 0 (cached_token_history empty,
    // reuse_cache disabled, has_images guard, or prefix mismatch).
    // Regardless of tokens.len(), the classifier routes to Miss so
    // the caller runs reset_caches_sync + init_caches_sync + full
    // prefill.
    assert_eq!(
        classify_prefix_cache_decision(0, 0),
        PrefixCacheDecision::Miss,
        "empty cache + empty tokens must be Miss"
    );
    assert_eq!(
        classify_prefix_cache_decision(0, 10),
        PrefixCacheDecision::Miss,
        "empty cache + non-empty tokens must be Miss"
    );
}

#[test]
fn strict_extend_is_hit() {
    // verify_cache_prefix returned cached_token_history.len() AND
    // tokens.len() > cached_token_history.len() — the new prompt
    // strictly extends the cached one. This is the only case that
    // takes the warm-reuse path: prefill_offset = cached_prefix_len,
    // so only the tail delta is prefilled.
    assert_eq!(
        classify_prefix_cache_decision(5, 8),
        PrefixCacheDecision::StrictExtendHit,
        "cached.len() < tokens.len() must be StrictExtendHit"
    );
    assert_eq!(
        classify_prefix_cache_decision(1, 2),
        PrefixCacheDecision::StrictExtendHit,
        "cached.len() = 1, tokens.len() = 2 must be StrictExtendHit (smallest hit)"
    );
}

#[test]
fn divergence_is_miss() {
    // verify_cache_prefix returned 0 because tokens[..cached.len()]
    // != cached[..] — semantically a divergence even though we only
    // observe the 0 return here. Same code path as `empty_cache_is_miss`
    // — both flavours of Miss fall into the same branch.
    assert_eq!(
        classify_prefix_cache_decision(0, 20),
        PrefixCacheDecision::Miss,
        "divergence (verifier returned 0) must be Miss"
    );
}

#[test]
fn exact_match_is_miss() {
    // verify_cache_prefix returned cached_token_history.len() AND
    // tokens.len() == cached_token_history.len() — byte-equal
    // prompt. The classifier routes to Miss because Gemma4 has no
    // snapshot of final-step logits and no safe "rewind by 1"
    // primitive over the sliding-window cache. Reprefilling the
    // last cached token over the live caches would advance cache
    // state to `prompt + last_token` (duplicated) while the
    // history write-back persists `tokens + generated`, desyncing
    // cache and history for the next warm-hit turn.
    //
    // This invariant guards against silently corrupting multi-turn
    // correctness.
    assert_eq!(
        classify_prefix_cache_decision(5, 5),
        PrefixCacheDecision::Miss,
        "exact-match (cached.len() == tokens.len()) must be Miss, not StrictExtendHit"
    );
    assert_eq!(
        classify_prefix_cache_decision(1, 1),
        PrefixCacheDecision::Miss,
        "exact-match single token must be Miss"
    );
    assert_eq!(
        classify_prefix_cache_decision(1000, 1000),
        PrefixCacheDecision::Miss,
        "exact-match long prompts must still be Miss"
    );
}

#[test]
fn invariant_cached_len_never_exceeds_tokens_len_in_hit() {
    // Belt-and-braces: the verifier itself returns 0 when
    // tokens.len() < cached.len() (no partial-cache reuse), so
    // `cached_prefix_len > tokens_len` should never be observed by
    // the classifier in practice. But if it ever was, the branch
    // routes it to Miss (cached_prefix_len < tokens_len is false),
    // which is the safe fallthrough.
    assert_eq!(
        classify_prefix_cache_decision(10, 5),
        PrefixCacheDecision::Miss,
        "cached_prefix_len > tokens_len must be Miss (defensive fallthrough)"
    );
}
