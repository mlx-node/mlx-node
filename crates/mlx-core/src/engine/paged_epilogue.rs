//! Paged-turn epilogue helpers shared by the single-adapter paged families.
//!
//! The verbatim pieces every pure-KV `PagedBackend` impl hand-copied before
//! this module existed:
//!   * [`SimplePagedPrefix`] — the two-usize prefix state (`qwen3`'s
//!     `Qwen3PrefixState`, k2's `K2PrefixState`, muse's `MusePrefixState`
//!     were byte-identical);
//!   * [`prime_single_adapter_prefix`] — the
//!     `prepare_turn_with_max_cache_hit_tokens` call + plan→state wrap;
//!   * [`finalize_single_adapter_turn`] / [`abort_single_adapter_turn`] —
//!     the `reuse → keep_live | register+release` / `release`-only
//!     lifecycle arms;
//!   * [`FinalTokenPolicy`] + [`save_paged_token_history`] /
//!     [`reconcile_paged_surplus`] — the `keep_all`/`drop-last` history trim
//!     and the `request_tokens` warm-continue rollback arithmetic.
//!
//! Families with extra per-turn state keep their own `PrefixState` (lfm2's
//! conv flag + `full_tokens`, nemotron_h's mamba flag, qwen3_5's GDN-prime
//! fields, gemma4's sliding-group fields); coordinator-backed families
//! (muse_glimmer, gemma4) share only the arithmetic — the `*_all` rollback
//! and `request_token_count_all` plug into [`reconcile_paged_surplus`]'s
//! closure/length parameters.

use crate::engine::backend::PagedPrefix;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

/// Two-length prefix state shared by every single-adapter paged family.
///
/// The EFFECTIVE cached-prefix / suffix split a `prepare_turn*` call
/// resolved (block-granular; the fresh suffix is always `>= 1` by the
/// vLLM cap). Families that referenced their own `*PrefixState` name in
/// scheduler paths keep it as a `type … = SimplePagedPrefix` alias.
pub(crate) struct SimplePagedPrefix {
    /// Effective cached-prefix length (block-granular). The fresh suffix
    /// the engine prefills is `tokens[effective_cached_prefix_len..]`.
    pub effective_cached_prefix_len: usize,
    /// Length of the fresh suffix prefilled this turn.
    pub suffix_len: usize,
}

impl PagedPrefix for SimplePagedPrefix {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

/// Prime a single-adapter prefix state — the
/// `prepare_turn_with_max_cache_hit_tokens` block at the head of every
/// pure-KV family's `prime_prefix_state` (qwen3's
/// `prime_prefix_state_for`, verbatim in k2).
///
/// Applies the vLLM-style exact-prefix cap (`total_budget - 1`: leave at
/// least one prompt token to prefill so the decoder always has something
/// to consume) and surfaces the resolved lengths in the returned state.
/// `skip_lookup` is `false` — text-only prefix lookup enabled.
pub(crate) fn prime_single_adapter_prefix(
    adapter: &mut PagedKVCacheAdapter,
    seq_id: SeqId,
    plan: &[u32],
    reuse_cache: bool,
    extra_keys: &[u64],
    cache_salt: u64,
) -> Result<SimplePagedPrefix, String> {
    let total_budget = plan.len() as u32;
    // vLLM-style exact-prefix cap: leave at least one prompt token to
    // prefill so the decoder always has something to consume.
    let max_cache_hit_tokens = total_budget.saturating_sub(1);
    let turn_plan = adapter.prepare_turn_with_max_cache_hit_tokens(
        seq_id,
        plan,
        total_budget,
        reuse_cache,
        extra_keys,
        cache_salt,
        false,
        max_cache_hit_tokens,
    )?;
    Ok(SimplePagedPrefix {
        effective_cached_prefix_len: turn_plan.cached_prefix_len as usize,
        suffix_len: turn_plan.suffix_len as usize,
    })
}

/// Success-path terminal lifecycle for a single-adapter paged turn:
/// `reuse_cache → finalize_turn_keep_live`, else
/// `register_full_blocks_for_reuse` + `release_request`.
///
/// The non-reuse arm attempts the release even when registration failed —
/// the `let _ =` call pairs this replaces ran `release_request`
/// unconditionally, and skipping it would strand the request live. The
/// first error (registration preferred) propagates so callers that want a
/// failure signal can see it; trait-level callers `let _ =` the result.
pub(crate) fn finalize_single_adapter_turn(
    adapter: &mut PagedKVCacheAdapter,
    reuse_cache: bool,
    extra_keys: &[u64],
    cache_salt: u64,
) -> Result<(), String> {
    if reuse_cache {
        adapter
            .finalize_turn_keep_live(extra_keys, cache_salt)
            .map(|_| ())
    } else {
        let register_result = adapter.register_full_blocks_for_reuse(extra_keys, cache_salt);
        let release_result = adapter.release_request();
        register_result.and(release_result).map(|_| ())
    }
}

/// Error-path teardown for a single-adapter paged turn: `release_request`
/// ONLY — never register / keep live (partial `block_table` state is
/// unsafe to keep).
pub(crate) fn abort_single_adapter_turn(adapter: &mut PagedKVCacheAdapter) -> Result<(), String> {
    adapter.release_request().map(|_| ())
}

/// Whether the paged decode loop forwards the final sampled token into
/// the cache — the knob that splits the shared history-trim arithmetic.
///
///   * [`Self::KeepAllOnLength`]: pure-KV families whose stepper
///     materializes the final token on a length exit
///     ([`crate::engine::backend::DecodeStep::materialize_final`]) —
///     `keep_all == (finish_reason == "length" && retain_final_length_token)`
///     keeps ALL generated tokens; any other stop drops the trailing
///     boundary token the next delta re-renders. (qwen3, k2, muse_glimmer,
///     gemma4.)
///   * [`Self::AlwaysDrop`]: conv/GDN/mamba families whose final token
///     never enters the cache — the saved history drops it ALWAYS and
///     `keep_all` is ignored. (lfm2, nemotron_h, qwen3_5, qwen3_5_moe.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinalTokenPolicy {
    KeepAllOnLength,
    AlwaysDrop,
}

impl FinalTokenPolicy {
    /// Number of `generated` tokens that belong in the saved history.
    ///
    /// == the `if keep_all || generated.is_empty() { len } else { len - 1 }`
    /// trim every family impl inlined (the `is_empty` guard keeps the
    /// drop-last from underflowing on a zero-token turn).
    pub(crate) fn history_len(self, generated_len: usize, keep_all: bool) -> usize {
        match self {
            Self::KeepAllOnLength if keep_all || generated_len == 0 => generated_len,
            Self::KeepAllOnLength => generated_len - 1,
            Self::AlwaysDrop => generated_len.saturating_sub(1),
        }
    }

    /// The `generated` tokens that belong in the saved history — the same
    /// trim as [`Self::history_len`], as a slice.
    pub(crate) fn history_tokens(self, generated: &[u32], keep_all: bool) -> &[u32] {
        &generated[..self.history_len(generated.len(), keep_all)]
    }
}

/// Shared `save_paged_history` arithmetic: `history =
/// save_tokens + trim(generated)` on the reuse path, `history.clear()`
/// otherwise.
///
/// Callers pass their own history vec (`self.cached_token_history`); any
/// extra per-family bookkeeping (qwen3's `cached_image_key = None`,
/// gemma4's media-session reset, the qwen3_5 GDN checkpoint) stays in the
/// impl around this call.
pub(crate) fn save_paged_token_history(
    save_tokens: &[u32],
    generated: &[u32],
    keep_all: bool,
    reuse_cache: bool,
    policy: FinalTokenPolicy,
    history: &mut Vec<u32>,
) {
    if reuse_cache {
        let mut full_history = save_tokens.to_vec();
        full_history.extend_from_slice(policy.history_tokens(generated, keep_all));
        *history = full_history;
    } else {
        history.clear();
    }
}

/// Shared `reconcile_paged_request_tokens` arithmetic: roll the recorded
/// token set back by `recorded_len - (prompt_len + history_len)` when that
/// surplus is positive, so `request_tokens()` matches the to-be-saved
/// history and the next turn's warm-continue gate is not defeated by a
/// trailing stop token the pipelined loop recorded at the loop top.
///
/// `recorded_len` is the family's recorded-token count (single-adapter
/// families pass `adapter.request_tokens().len()`; coordinator families
/// pass `request_token_count_all(seq_id)`); `rollback` is the matching
/// rollback fn (`adapter.rollback_last_tokens` /
/// `coordinator.rollback_last_tokens_all`).
///
/// `Ok(())` covers both reconciled and no-op (surplus 0). `Err((surplus,
/// source))` surfaces a failed rollback WITH the attempted count, so the
/// caller's `reconcile_paged_request_tokens` can keep its own warn
/// (per-family `target:` is a const callsite and cannot live here) and
/// return `false` — the engine then finalizes with `reuse_cache = false`
/// (`release_request`, not keep-live). For single-adapter callers
/// `surplus <= recorded_len`, so the rollback's `n > len` `Err` cannot
/// fire; coordinator callers measure `recorded_len` on the FULL adapter
/// while the rollback spans all groups, so the `Err` arm is a real
/// defensive contract there, not just unreachable.
pub(crate) fn reconcile_paged_surplus(
    recorded_len: usize,
    prompt_len: usize,
    generated_len: usize,
    keep_all: bool,
    policy: FinalTokenPolicy,
    rollback: impl FnOnce(u32) -> Result<(), String>,
) -> Result<(), (usize, String)> {
    // `history_len` uses the EXACT same trim as `save_paged_token_history`
    // (same `keep_all`, same policy), so the two never disagree.
    let history_len = policy.history_len(generated_len, keep_all);
    let target_len = prompt_len + history_len;
    let surplus = recorded_len.saturating_sub(target_len);
    if surplus > 0 {
        rollback(surplus as u32).map_err(|e| (surplus, e))?;
    }
    Ok(())
}
