//! Bounded retention for materialized GDN recurrent-state checkpoints.
//!
//! Two consumers read this store and they want opposite things. A warm turn
//! wants the DEEPEST checkpoint on its lineage. A cold-tier sidecar capture
//! wants the deepest checkpoint that is still at or below the prefix the
//! persisted K/V chain has actually reached, and that chain lags the prompt
//! badly for the first several turns — so it wants a SHALLOW one.
//!
//! A prefill serves both by publishing a ladder of rungs
//! (`paged_forward::gdn_prefill_checkpoint_boundaries`), but only turn 1
//! publishes a full ladder.
//!
//! (F2) `gdn_prefill_checkpoint_boundaries` drops every rung at or below
//! `cached_prefix_len` because this prefill never crosses those token positions,
//! so a warm turn on a mostly cached prompt publishes its deepest rung alone —
//! pinned by `paged_forward::tests::ladder_rungs_are_quarters_of_the_one_above`,
//! where a 4096-token prompt with 1008 cached yields `[4080]`.
//!
//! (F1) A turn that finds no usable checkpoint re-forwards its whole cached
//! prefix through the GDN layers and does NOT reset `cached_prefix_len`
//! (`model.rs`'s `prepare_dense_gdn_prefix_state`, `replay_materialized` arm),
//! so the paged prefix it re-forwards is the full one, every time it happens.
//!
//! Together these say the shallow rungs a later capture anchors on exist ONLY as
//! residue from an earlier turn: an owner that loses them pays a full re-forward
//! on its next turn and gets back only its deepest rung. Retention is not a
//! nicety here — evicting a shallow rung destroys coverage nothing will
//! republish until the conversation is reset.

use std::collections::VecDeque;

/// One resident GDN sidecar for the root Pi session plus the four child loops
/// the subagent extension can run concurrently. Sidecars are large (about
/// 75 MiB for the dense 27B checkpoint), so keep this a hard bound rather than
/// making it scale with the paged KV pool or the eight-task submission limit.
///
/// This number was chosen when an owner needed ONE checkpoint. It no longer
/// covers what an owner needs: a warm turn wants that owner's deepest rung and
/// a cold capture wants a shallow one, so every concurrent owner now wants two
/// slots, and the owner publishing right now wants
/// `GDN_PREFIX_CHECKPOINTS_PER_OWNER` of them. Five slots therefore fit at most
/// two live owners with full coverage, and a publisher taking four leaves
/// exactly one. That shortfall — not the eviction order — is what
/// `prune_gdn_checkpoints` is rationing; see the characterization tests named
/// `four_single_entry_siblings_*` and `one_subagent_turn_*`.
///
/// The cliff sits exactly at `distinct owners > limit`, where every owner holds
/// one entry and the store is still over cap, so each publish takes somebody's
/// last checkpoint. `retention_sim`'s 40-turn sweep at six owners measures 28
/// turns with no checkpoint at all and 84% of every cached prefix re-forwarded,
/// under BOTH global-cap arms; at five owners it measures 0 and 10.5%. Six
/// slots would move the cliff to seven owners for +75 MiB of resident 27B
/// checkpoints (`retention_sim`'s capacity sweep prints the row). It is not
/// taken because the subagent extension caps concurrency at four child loops
/// plus the root, which is five. A `/new` rotation mints a sixth owner id, but
/// the retired root is the one `last_resort_victim` takes first — pinned by
/// `rotating_root_evicts_old_root_and_preserves_new_root_with_four_children`.
///
/// What the cap DOES cost at the sized fleet is SSD writes a restart would read,
/// not restore depth: at five owners the store cannot hold every owner's whole
/// ladder, so a shallow rung an owner just published can be evicted before it is
/// written. That shortfall is a bounded miss window, not a permanent loss —
/// the persisted chain gains on the prompt (+`QUEUE_BLOCKS_PER_TURN` blocks a
/// turn against the fixture's per-turn prompt growth), so the deepest retained
/// rung comes under the chain's reach within a few turns and every owner ends
/// up with the same deepest sidecar on disk. That bound holds ONLY because the
/// writer is fast enough for the chain to gain; at the pre-`fsync(2)` rate the
/// chain LOST ground per turn and the shortfall was permanent. So a writer
/// regression reads here as a capacity regression — `retention_sim` sweeps
/// `PRE_FSYNC_BLOCKS_PER_TURN` for exactly that reason.
///
/// The fleet size this is paired with lives in another language, so the pairing
/// is a gate rather than a paragraph: `cold_tier::gdn_prefix_checkpoint_limit`
/// publishes this number and
/// `packages/agent/__test__/gdn-checkpoint-capacity.test.ts` holds
/// `MAX_CONCURRENCY + 1` against it. Every Rust gate here is blind to a
/// TypeScript-only edit — `retention_sim`'s sweep bound is a fixed 5, not this
/// cap, deliberately — so that test is the only thing standing between a raised
/// `MAX_CONCURRENCY` and the six-owner cliff described above.
pub(crate) const GDN_PREFIX_CHECKPOINT_LIMIT: usize = 5;

/// One prefill publishes a ladder of block-aligned prefix checkpoints rather
/// than a single endpoint, because a cold sidecar can only anchor on the part
/// of the K/V chain that actually reached disk, which lags the prompt for the
/// first several turns. This cap is the ladder width
/// (`GDN_CHECKPOINT_LADDER_RUNGS`), so this cap alone never trims a ladder the
/// turn that published it still needs. The GLOBAL cap does, whenever the store
/// holds other owners — see `prune_gdn_checkpoints`.
///
/// It applies ONLY to a turn that publishes a ladder. A turn with no cold GDN
/// policy publishes one rung, and widening its store instead changes which
/// checkpoint later turns land on — see
/// [`GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER`]. Pick between the two through
/// [`gdn_retention_caps`], never by naming one directly at a call site.
pub(crate) const GDN_PREFIX_CHECKPOINTS_PER_OWNER: usize = 4;

/// Per-owner retention for a turn that publishes no ladder. This is the
/// pre-ladder cap, restored verbatim.
///
/// The wider cap is not free on that arm, and the cost is emitted tokens, not
/// memory. With no cold GDN policy a prefill publishes ONE boundary
/// (`paged_forward::prefill_checkpoint_boundaries`, `want_ladder = false`), so
/// the extra slots retain whole SIBLING lineages rather than a publisher's own
/// shallow rungs — several conversations multiplexed over one model all publish
/// under the same implicit owner id. Whether a later turn of the FIRST
/// conversation still finds its own entry therefore depends on this number, and
/// the two outcomes build the GDN recurrent state by different code:
/// `prepare_dense_gdn_prefix_state`'s `checkpoint` arm installs a snapshot the
/// chunked prefill took, its `replay_materialized` arm re-forwards the whole
/// cached prefix through `run_gdn_only_prefill_materialized`. Different span,
/// different accumulation order, different tokens.
///
/// Measured on real weights (qwen3.5-0.8b-mlx-bf16, greedy, persistence OFF,
/// `MLX_PAGED_PREFILL_CHUNK_SIZE=2048` as `mlx launch claude` sets it, three
/// conversations over one model, turn 2 of the first):
///
/// ```text
///   cap 4   state=checkpoint           restored 3584  replayed    0
///   cap 2   state=replay_materialized  restored    0  replayed 3584
///           -> emitted text diverges at character 56 of the same prompt
/// ```
///
/// So this is the same contract `prefill_checkpoint_boundaries` keeps: a
/// persistence-OFF request must take the retention it took before the ladder
/// existed. Persist-ON pays the drift knowingly, in exchange for a restorable
/// prefix; persist-OFF gets nothing back for it.
pub(crate) const GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER: usize = 2;

/// Which victim a [`prune_gdn_checkpoints`] eviction picks, when more than one
/// entry is eligible.
///
/// The two caps decide HOW MANY entries survive; this decides WHICH. Both move
/// the same observable quantity — the depth a later turn restores from — so both
/// have to answer to the same `want_ladder` predicate. See
/// [`GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER`] for why a persistence-OFF turn
/// may not pay for either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GdnRetentionPolicy {
    /// The victim order this store used before the checkpoint ladder existed:
    /// the FIRST same-owner ancestor in publish order, then the first entry of
    /// an owner holding a spare, then the first non-root entry.
    ///
    /// Restored verbatim for turns that publish no ladder, so a persistence-OFF
    /// session keeps not just the pre-ladder cap but the pre-ladder retained
    /// SET. With one rung published per turn, "first ancestor in publish order"
    /// is the shallowest rung, so a monotone conversation keeps its DEEPEST
    /// entries — which is what a warm turn with a short paged hit lands on.
    PreLadder,
    /// Ladder-aware: defer the publishing owner (its cold capture runs at the
    /// end of this same turn), then drop the most redundant rung by
    /// [`least_valuable_same_owner_ancestor`]'s ratio.
    ///
    /// This deliberately keeps SHALLOW rungs alive, because those are the only
    /// ones a cold restore can anchor on while the persisted K/V chain still
    /// lags the prompt. That trade only pays for a turn that writes a sidecar.
    Ladder,
}

/// Everything one `prune_gdn_checkpoints` call enforces.
///
/// ```text
///   want_ladder  explicit_root   ->  limit  per_owner  policy
///   false        false               2      2          PreLadder
///   false        true                5      2          PreLadder
///   true         false               4      4          Ladder
///   true         true                5      4          Ladder
/// ```
///
/// `want_ladder` is `paged_forward::gdn_cold_sidecar_ladder_wanted` for the
/// turn's adapter — the same predicate that decides the break set — so
/// retention and the published rungs cannot disagree about whether this is a
/// persist turn. `explicit_root` is `gdn_root_cache_owner_is_explicit`: without
/// a named root there is no separate global budget to spend, so the per-owner
/// cap is also the global one.
///
/// The predicate may flip mid-session (a cold tier is installed after some
/// turns have run). That is safe in both directions and needs no reconciliation:
/// 4 -> 2 only prunes down at the next publish, and 2 -> 4 cannot resurrect an
/// entry that is already gone.
///
/// Single seam on purpose. The dense and MoE call sites are duplicated code, and
/// this is the one decision they must not drift on — and the policy travels with
/// the caps for the same reason, so no call site can take one from the ladder
/// column and the other from the pre-ladder column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GdnRetentionCaps {
    /// Total entries the store may hold.
    pub(crate) limit: usize,
    /// Entries any one owner may hold.
    pub(crate) per_owner: usize,
    /// Which eligible entry an eviction takes.
    pub(crate) policy: GdnRetentionPolicy,
}

#[cfg(test)]
impl GdnRetentionCaps {
    /// Caps with the ladder victim order, for tests that pin a bound directly
    /// rather than going through [`gdn_retention_caps`].
    pub(crate) fn ladder(limit: usize, per_owner: usize) -> Self {
        Self {
            limit,
            per_owner,
            policy: GdnRetentionPolicy::Ladder,
        }
    }

    /// Caps with the pre-ladder victim order.
    pub(crate) fn pre_ladder(limit: usize, per_owner: usize) -> Self {
        Self {
            limit,
            per_owner,
            policy: GdnRetentionPolicy::PreLadder,
        }
    }
}

pub(crate) fn gdn_retention_caps(want_ladder: bool, explicit_root: bool) -> GdnRetentionCaps {
    let (per_owner, policy) = if want_ladder {
        (GDN_PREFIX_CHECKPOINTS_PER_OWNER, GdnRetentionPolicy::Ladder)
    } else {
        (
            GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
            GdnRetentionPolicy::PreLadder,
        )
    };
    let limit = if explicit_root {
        GDN_PREFIX_CHECKPOINT_LIMIT
    } else {
        per_owner
    };
    GdnRetentionCaps {
        limit,
        per_owner,
        policy,
    }
}

/// Keep one slot the publishing owner cannot occupy by itself. Equalize the two
/// and a single owner at its own cap fills the store, so the first checkpoint
/// any second owner publishes is already over the global cap and every later
/// turn trades one entry for one entry.
///
/// This constrains the constants, not the call sites. On the implicit-root path
/// [`gdn_retention_caps`] hands the per-owner cap over as the global limit too,
/// because there is no named root to budget siblings against.
const _: () = assert!(GDN_PREFIX_CHECKPOINTS_PER_OWNER < GDN_PREFIX_CHECKPOINT_LIMIT);

/// The pre-ladder cap can only ever be the tighter of the two. Reversing them
/// would make a persistence-OFF turn retain MORE than a persist turn, which is
/// the opposite of what the gate exists to do.
const _: () =
    assert!(GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER <= GDN_PREFIX_CHECKPOINTS_PER_OWNER);

/// The ladder cap IS the ladder width; the doc on
/// [`GDN_PREFIX_CHECKPOINTS_PER_OWNER`] says so and nothing else enforced it.
/// Widening the ladder alone would silently trim every ladder by the difference
/// on the very push that produced it.
const _: () = assert!(
    GDN_PREFIX_CHECKPOINTS_PER_OWNER == super::paged_forward::GDN_CHECKPOINT_LADDER_RUNGS as usize
);

/// Minimal identity needed to decide whether two materialized GDN sidecars
/// belong to the same exact paged-cache lineage.
///
/// The block-hash chain is load-bearing: token ancestry alone is insufficient
/// when VLM extra keys or a cache salt differ. Each hash already commits to the
/// preceding hash, the block's tokens, and that block's extra keys (including
/// the first-block salt), so a chain prefix is the same identity used by the
/// paged allocator itself.
pub(crate) trait GdnCheckpointLineage {
    fn owner_id(&self) -> &str;
    fn prefix_len(&self) -> u32;
    fn block_size(&self) -> u32;
    fn final_block_hash(&self) -> u64;
    fn tokens(&self) -> &[u32];
    fn block_hashes(&self) -> &[u64];
}

/// Replay into a staged GDN cache and publish it only after the replay succeeds.
///
/// Materialized replay is fallible after it has already mutated recurrent
/// state. Keeping that state local prevents a failed prefix preparation from
/// leaving the model's active cache partially advanced.
pub(crate) fn replay_gdn_cache_and_commit<T, E, F>(
    active: &mut Option<T>,
    mut staged: T,
    replay: F,
) -> Result<(), E>
where
    F: FnOnce(&mut T) -> Result<(), E>,
{
    replay(&mut staged)?;
    *active = Some(staged);
    Ok(())
}

fn is_strict_ancestor<T: GdnCheckpointLineage>(ancestor: &T, descendant: &T) -> bool {
    ancestor.block_size() == descendant.block_size()
        && ancestor.tokens().len() < descendant.tokens().len()
        && ancestor.block_hashes().len() < descendant.block_hashes().len()
        && descendant.tokens().starts_with(ancestor.tokens())
        && descendant
            .block_hashes()
            .starts_with(ancestor.block_hashes())
}

fn is_same_owner_ancestor<T: GdnCheckpointLineage>(ancestor: &T, descendant: &T) -> bool {
    ancestor.owner_id() == descendant.owner_id() && is_strict_ancestor(ancestor, descendant)
}

/// Which owners' checkpoints an eviction search is allowed to take from.
///
/// `Excluding` is what defers a turn's own ladder: the owner publishing right
/// now is the one whose capture runs at the end of this same turn, so its rungs
/// are the last thing worth dropping even though they are also the
/// cheapest-looking (a ladder is nothing but same-owner ancestors, which the
/// ancestor-first policy would otherwise eat one rung per push). It defers, it
/// does not protect — when no other owner holds a spare rung the search comes
/// back empty and the ladder is eaten anyway.
#[derive(Clone, Copy)]
enum OwnerScope<'a> {
    Any,
    Only(&'a str),
    Excluding(&'a str),
}

impl OwnerScope<'_> {
    fn admits(self, owner_id: &str) -> bool {
        match self {
            OwnerScope::Any => true,
            OwnerScope::Only(only) => owner_id == only,
            OwnerScope::Excluding(excluded) => owner_id != excluded,
        }
    }
}

/// Index of the same-owner ancestor whose eviction costs the least prefix
/// coverage.
///
/// The candidate set is exactly the one an ancestor-first policy uses: an entry
/// that is a strict prefix of another entry with the same owner. Ancestors are
/// the cheapest thing to drop because the in-memory lookup always takes the
/// longest valid checkpoint at or below what it asked for, so a deeper entry on
/// the same lineage already answers everything its own ancestors could. A
/// distinct lineage answers nothing else and stays protected.
///
/// Which ancestor to drop is decided by how close it sits below its next-deeper
/// sibling, as a ratio so the choice does not depend on absolute prompt length.
/// On a geometrically spaced ladder that is the most redundant rung, and the
/// shallow rungs survive — those are the only ones a cold restore can anchor on
/// while the persisted K/V chain still lags the prompt. Ties prefer the deeper
/// rung (coverage is scarcer at the shallow end, which has nothing below it),
/// then the least recently published entry.
fn least_valuable_same_owner_ancestor<T: GdnCheckpointLineage>(
    checkpoints: &VecDeque<T>,
    scope: OwnerScope<'_>,
) -> Option<usize> {
    let mut best: Option<(usize, u64, u64)> = None;
    for idx in 0..checkpoints.len() {
        let candidate = &checkpoints[idx];
        if !scope.admits(candidate.owner_id()) {
            continue;
        }
        let has_descendant = checkpoints
            .iter()
            .enumerate()
            .any(|(other_idx, other)| other_idx != idx && is_same_owner_ancestor(candidate, other));
        if !has_descendant {
            continue;
        }
        let own_len = u64::from(candidate.prefix_len());
        if own_len == 0 {
            continue;
        }
        let Some(next_len) = checkpoints
            .iter()
            .filter(|other| other.owner_id() == candidate.owner_id())
            .map(|other| u64::from(other.prefix_len()))
            .filter(|len| *len > own_len)
            .min()
        else {
            continue;
        };
        let preferred = match best {
            None => true,
            // `next_len / own_len < best_next / best_own`, cross-multiplied.
            Some((_, best_next, best_own)) => {
                match (next_len * best_own).cmp(&(best_next * own_len)) {
                    std::cmp::Ordering::Less => true,
                    std::cmp::Ordering::Greater => false,
                    std::cmp::Ordering::Equal => own_len > best_own,
                }
            }
        };
        if preferred {
            best = Some((idx, next_len, own_len));
        }
    }
    best.map(|(idx, _, _)| idx)
}

/// Cheapest entry to drop whose owner still keeps a checkpoint afterwards:
/// a redundant rung first, then any entry from an owner holding more than one.
///
/// Every eviction this returns costs prefix COVERAGE and nothing else. No owner
/// loses warm reuse, because whoever pays still has an entry the next lookup can
/// land on. That is the whole reason this is separated from
/// [`last_resort_victim`], which is not free in that sense and must never be
/// reached while this still has an answer.
fn redundant_victim<T: GdnCheckpointLineage>(
    checkpoints: &VecDeque<T>,
    scope: OwnerScope<'_>,
) -> Option<usize> {
    least_valuable_same_owner_ancestor(checkpoints, scope)
        .or_else(|| owner_with_a_spare(checkpoints, scope))
}

/// First entry, in publish order, whose owner would still hold a checkpoint
/// after it is dropped.
///
/// Shared by both policies: it is the second arm of [`redundant_victim`] and,
/// verbatim, the second arm of the pre-ladder global search.
fn owner_with_a_spare<T: GdnCheckpointLineage>(
    checkpoints: &VecDeque<T>,
    scope: OwnerScope<'_>,
) -> Option<usize> {
    (0..checkpoints.len()).find(|&idx| {
        let owner = checkpoints[idx].owner_id();
        scope.admits(owner)
            && checkpoints
                .iter()
                .filter(|checkpoint| checkpoint.owner_id() == owner)
                .count()
                > 1
    })
}

/// First same-owner ancestor in publish order — the pre-ladder choice.
///
/// Same candidate set as [`least_valuable_same_owner_ancestor`], different tie
/// break: position instead of redundancy ratio. On a monotone lineage that is
/// the SHALLOWEST rung, so repeated pruning keeps the deepest entries; the
/// ratio rule keeps a shallow one on purpose. The two therefore answer a later
/// lookup from different depths, which is why the choice is gated rather than
/// improved in place.
fn first_same_owner_ancestor<T: GdnCheckpointLineage>(
    checkpoints: &VecDeque<T>,
    scope: OwnerScope<'_>,
) -> Option<usize> {
    (0..checkpoints.len()).find(|&idx| {
        scope.admits(checkpoints[idx].owner_id())
            && checkpoints.iter().enumerate().any(|(other_idx, other)| {
                other_idx != idx && is_same_owner_ancestor(&checkpoints[idx], other)
            })
    })
}

/// Least-recent entry that is the LAST one its owner holds, skipping the root.
///
/// This is the only eviction that takes an owner's warm reuse away rather than
/// just narrowing it. By (F1) the owner's very next turn then re-forwards its
/// whole cached prefix through the GDN layers — the expensive part — and by (F2)
/// that turn republishes its DEEPEST rung only, so the shallow coverage it lost
/// does not come back for the rest of the conversation and its later cold
/// captures miss while the persisted chain still lags. It is not blind forever
/// (the republished deepest rung answers the turn after), but each trip through
/// here costs one full re-forward and permanently narrows that owner.
///
/// Reaching this arm at all means the store holds one entry per owner and is
/// still over cap, i.e. there are more live owners than slots and somebody has
/// to lose. Nothing above may fall through to it early.
///
/// Returns `None` when the only entries left are the root's.
fn last_resort_victim<T: GdnCheckpointLineage>(
    checkpoints: &VecDeque<T>,
    scope: OwnerScope<'_>,
    root_owner_id: &str,
) -> Option<usize> {
    (0..checkpoints.len()).find(|&idx| {
        let owner = checkpoints[idx].owner_id();
        scope.admits(owner) && owner != root_owner_id
    })
}

/// Enforce the bounded branch-aware retention policy.
///
/// `caps.policy` selects the victim order, and everything below describes
/// [`GdnRetentionPolicy::Ladder`] — the arm a turn that publishes a ladder
/// takes. [`GdnRetentionPolicy::PreLadder`] restores 77e43031's order verbatim
/// for turns that publish none: first same-owner ancestor in publish order, then
/// [`owner_with_a_spare`], then [`last_resort_victim`], with no deferral of the
/// publishing owner. Both caps AND the victim order are gated together, because
/// both decide the depth a later turn restores from, and a persistence-OFF turn
/// buys nothing with that drift.
///
/// First enforce the owner cap, preferring the ancestor
/// `least_valuable_same_owner_ancestor` picks.
///
/// Then enforce the global cap, in three arms, and the order of the first two
/// is the whole of what `active_owner_id` decides:
///
/// 1. a redundant entry belonging to anyone EXCEPT `active_owner_id`,
/// 2. a redundant entry belonging to anyone at all — in practice the publisher's
///    own, because arm 1 has already taken everyone else's,
/// 3. the least-recent entry that is the LAST one its owner holds, skipping the
///    root.
///
/// Arms 1 and 2 draw from one candidate set: [`redundant_victim`] never returns
/// an entry whose owner would be left with nothing. They differ only in who pays
/// first. Arm 3 is the only eviction that takes an owner's warm reuse away, and
/// it is reachable only when [`redundant_victim`] over `Any` is empty — every
/// owner already holds exactly one entry and the store is STILL over cap, i.e.
/// strictly more live owners than slots. Splitting the free set in two cannot
/// change when that happens: each iteration removes exactly one entry, and the
/// number of free evictions available is `len - distinct_owners` under either
/// order. So the ordering choice cannot blind an owner the unrestricted search
/// would have kept, and `retention_sim::the_publisher_arm_never_costs_hot_path_replay`
/// measures that over 40-turn conversations — identical replayed prefix tokens
/// and identical blind turns in every cell of the sweep.
///
/// What arm 1 buys is the publishing owner's SHALLOW rungs. `prune` runs after
/// EVERY rung a prefill pushes, not once per ladder, so a store already at the
/// cap sees the ladder arrive one rung at a time, and judged on redundancy alone
/// the newest rung's own predecessor is always the most redundant entry in the
/// store. Unrestricted, the ladder eats itself: `[48, 240, 1008, 4080]`
/// published over a full store retains only `4080`, exactly the single-endpoint
/// behaviour the ladder exists to replace, and the sidecar capture that runs at
/// the end of that same turn has nothing under the persisted chain's reach.
/// `retention_sim::the_publisher_arm_buys_cold_sidecars` measures what deferring
/// the publisher is worth.
///
/// Arm 1 is a preference, not a guarantee. When no other owner holds redundancy
/// — four siblings with one checkpoint each, the shape
/// `GDN_PREFIX_CHECKPOINT_LIMIT` was sized for — it finds nothing and arm 2 eats
/// the publisher's ladder anyway; `four_single_entry_siblings_outlive_the_publishers_ladder`
/// pins that. It is the right trade rather than a regression: a missed sidecar
/// costs one SSD write that only a restart would have read, while a sibling
/// pushed to zero re-forwards its whole cached prefix through the GDN layers on
/// its next turn AND never republishes a shallow rung, by (F1) and (F2) in the
/// module doc. It is still a real cost, and it is why five slots is a shortfall
/// rather than a fit.
///
/// Two floors hold: the root's last checkpoint survives any ladder any child
/// publishes (arm 3 skips it), and no owner is reduced below one checkpoint
/// while the store holds at most `limit` distinct owners
/// (`one_subagent_turn_leaves_every_sibling_a_checkpoint`). Past `limit` owners
/// somebody has to lose, both arms alike; `retention_sim`'s six-owner rows are
/// what that looks like.
pub(crate) fn prune_gdn_checkpoints<T: GdnCheckpointLineage>(
    checkpoints: &mut VecDeque<T>,
    caps: GdnRetentionCaps,
    root_owner_id: &str,
    active_owner_id: &str,
) {
    let owners: Vec<String> = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.owner_id().to_owned())
        .collect();
    for owner in owners {
        while checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.owner_id() == owner)
            .count()
            > caps.per_owner
        {
            let scope = OwnerScope::Only(&owner);
            let ancestor = match caps.policy {
                GdnRetentionPolicy::PreLadder => first_same_owner_ancestor(checkpoints, scope),
                GdnRetentionPolicy::Ladder => {
                    least_valuable_same_owner_ancestor(checkpoints, scope)
                }
            };
            let victim = ancestor.or_else(|| {
                (0..checkpoints.len()).find(|&idx| checkpoints[idx].owner_id() == owner)
            });
            let Some(idx) = victim else {
                break;
            };
            checkpoints.remove(idx);
        }
    }

    while checkpoints.len() > caps.limit {
        let victim = match caps.policy {
            // Pre-ladder: no deferral of the publishing owner, and position
            // rather than ratio decides among ancestors.
            GdnRetentionPolicy::PreLadder => {
                first_same_owner_ancestor(checkpoints, OwnerScope::Any)
                    .or_else(|| owner_with_a_spare(checkpoints, OwnerScope::Any))
            }
            GdnRetentionPolicy::Ladder => {
                redundant_victim(checkpoints, OwnerScope::Excluding(active_owner_id))
                    .or_else(|| redundant_victim(checkpoints, OwnerScope::Any))
            }
        }
        .or_else(|| last_resort_victim(checkpoints, OwnerScope::Any, root_owner_id))
        .unwrap_or(0);
        checkpoints.remove(victim);
    }
}

/// Compute every allocator-compatible block hash through `prefix_len`.
pub(crate) fn compute_paged_prefix_block_hashes(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
) -> Option<Vec<u64>> {
    if prefix_len == 0 || block_size == 0 || !prefix_len.is_multiple_of(block_size) {
        return None;
    }

    let prefix_len = prefix_len as usize;
    let block_size = block_size as usize;
    if prefix_len > tokens.len() {
        return None;
    }

    let num_blocks = prefix_len / block_size;
    let mut hashes = Vec::with_capacity(num_blocks);
    let mut parent_hash = 0;
    for block_idx in 0..num_blocks {
        let extra_keys = extra_keys_per_block.get(block_idx)?;
        let start = block_idx * block_size;
        let end = start + block_size;
        parent_hash = if block_idx == 0 && cache_salt != 0 {
            let mut salted_keys = Vec::with_capacity(extra_keys.len() + 1);
            salted_keys.extend_from_slice(extra_keys);
            salted_keys.push(cache_salt);
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, &salted_keys)
        } else {
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, extra_keys)
        };
        hashes.push(parent_hash);
    }

    Some(hashes)
}

#[cfg(test)]
pub(crate) fn compute_paged_prefix_block_hash(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
) -> Option<u64> {
    compute_paged_prefix_block_hashes(
        tokens,
        prefix_len,
        block_size,
        extra_keys_per_block,
        cache_salt,
    )
    .and_then(|hashes| hashes.last().copied())
}

/// Find the most recent longest checkpoint on the requested paged-cache
/// lineage.
///
/// PagedAttention may back a cache hit up by one or more complete blocks when
/// the live tail cannot be reused. A sidecar captured at the former live head
/// is then a descendant of the requested prefix and cannot be restored, but an
/// older retained sidecar can still be a valid ancestor. Match that ancestor
/// using the allocator's complete hash chain (which commits to tokens, cache
/// salt, and per-block extra keys), then let the caller replay only the gap.
pub(crate) fn find_longest_valid_gdn_checkpoint_index<T, F>(
    checkpoints: &VecDeque<T>,
    owner_id: &str,
    tokens: &[u32],
    requested_prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
    mut caches_ready: F,
) -> Option<usize>
where
    T: GdnCheckpointLineage,
    F: FnMut(&T) -> bool,
{
    let requested_block_hashes = compute_paged_prefix_block_hashes(
        tokens,
        requested_prefix_len,
        block_size,
        extra_keys_per_block,
        cache_salt,
    )?;

    let mut best: Option<(usize, u32)> = None;
    for (idx, checkpoint) in checkpoints.iter().enumerate() {
        let prefix_len = checkpoint.prefix_len();
        let prefix_len_usize = prefix_len as usize;
        let expected_blocks = prefix_len.checked_div(block_size)? as usize;
        let valid = checkpoint.owner_id() == owner_id
            && prefix_len > 0
            && prefix_len <= requested_prefix_len
            && prefix_len.is_multiple_of(block_size)
            && checkpoint.block_size() == block_size
            && checkpoint.tokens().len() == prefix_len_usize
            && tokens.get(..prefix_len_usize) == Some(checkpoint.tokens())
            && checkpoint.block_hashes().len() == expected_blocks
            && requested_block_hashes.get(..expected_blocks) == Some(checkpoint.block_hashes())
            && checkpoint.block_hashes().last().copied() == Some(checkpoint.final_block_hash())
            && caches_ready(checkpoint);
        if !valid {
            continue;
        }

        // The queue is LRU ordered, so prefer the later entry when duplicate
        // lineage lengths exist.
        if best.is_none_or(|(best_idx, best_len)| {
            prefix_len > best_len || (prefix_len == best_len && idx > best_idx)
        }) {
            best = Some((idx, prefix_len));
        }
    }
    best.map(|(idx, _)| idx)
}

/// What the retained store looked like when a cold-tier sidecar capture found
/// nothing to write.
///
/// `deepest_retained` is the diagnostic that separates the two very different
/// causes of an empty result: no checkpoint on this lineage at all, versus a
/// checkpoint that exists but sits past the persisted K/V chain's reach. Only
/// the second is a policy problem, and without this field both look identical
/// from outside.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GdnSidecarBoundaryMiss {
    pub deepest_retained: Option<u32>,
    pub retained_for_owner: usize,
    pub retained_total: usize,
}

/// Outcome of choosing where a GDN cold sidecar may be anchored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnSidecarBoundary {
    Selected { index: usize, boundary: u32 },
    Missed(GdnSidecarBoundaryMiss),
}

/// Choose the deepest retained checkpoint a cold sidecar may be written at.
///
/// A GDN recurrent state is valid ONLY at the exact prefix length it was
/// produced at, so the boundary is never computed — it is picked from the
/// checkpoints a prefill already materialized. `chain_reach_tokens` caps the
/// choice at the prefix the persisted K/V chain covers: a sidecar past that
/// frontier could never be selected on restore, so writing one only burns
/// quota.
///
/// On a miss the returned `GdnSidecarBoundaryMiss` describes the store the
/// decision was made against; callers pair it with their own request context
/// in a trace.
pub(crate) fn select_gdn_sidecar_boundary<T, F>(
    checkpoints: &VecDeque<T>,
    owner_id: &str,
    tokens: &[u32],
    chain_reach_tokens: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
    caches_ready: F,
) -> GdnSidecarBoundary
where
    T: GdnCheckpointLineage,
    F: FnMut(&T) -> bool,
{
    if let Some(index) = find_longest_valid_gdn_checkpoint_index(
        checkpoints,
        owner_id,
        tokens,
        chain_reach_tokens,
        block_size,
        extra_keys_per_block,
        cache_salt,
        caches_ready,
    ) {
        let boundary = checkpoints[index].prefix_len();
        // The search already accepts only `0 < prefix_len <= chain_reach_tokens`.
        // Re-checking here keeps that invariant enforced in one place, so a
        // future loosening of the search cannot silently anchor a sidecar the
        // restore walk could never select.
        if boundary > 0 && boundary <= chain_reach_tokens {
            return GdnSidecarBoundary::Selected { index, boundary };
        }
    }

    GdnSidecarBoundary::Missed(GdnSidecarBoundaryMiss {
        deepest_retained: checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.owner_id() == owner_id)
            .map(|checkpoint| checkpoint.prefix_len())
            .max(),
        retained_for_owner: checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.owner_id() == owner_id)
            .count(),
        retained_total: checkpoints.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    mod retention_sim;

    #[test]
    fn replay_commit_is_transactional() {
        let mut active = Some(vec![1, 2]);
        let failed: Result<(), &'static str> =
            replay_gdn_cache_and_commit(&mut active, vec![10], |staged| {
                staged.push(11);
                Err("replay failed")
            });
        assert_eq!(failed, Err("replay failed"));
        assert_eq!(active, Some(vec![1, 2]));

        replay_gdn_cache_and_commit(&mut active, vec![20], |staged| {
            staged.push(21);
            Ok::<_, &'static str>(())
        })
        .unwrap();
        assert_eq!(active, Some(vec![20, 21]));
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Lineage {
        name: &'static str,
        owner: &'static str,
        prefix_len: u32,
        block_size: u32,
        final_block_hash: u64,
        tokens: Vec<u32>,
        hashes: Vec<u64>,
    }

    impl GdnCheckpointLineage for Lineage {
        fn owner_id(&self) -> &str {
            self.owner
        }

        fn prefix_len(&self) -> u32 {
            self.prefix_len
        }

        fn block_size(&self) -> u32 {
            self.block_size
        }

        fn final_block_hash(&self) -> u64 {
            self.final_block_hash
        }

        fn tokens(&self) -> &[u32] {
            &self.tokens
        }

        fn block_hashes(&self) -> &[u64] {
            &self.hashes
        }
    }

    fn lineage(name: &'static str, owner: &'static str, hashes: &[u64]) -> Lineage {
        Lineage {
            name,
            owner,
            prefix_len: hashes.len() as u32,
            block_size: 16,
            final_block_hash: hashes.last().copied().unwrap_or_default(),
            tokens: (0..hashes.len() as u32).collect(),
            hashes: hashes.to_vec(),
        }
    }

    /// Everything `prune_gdn_checkpoints` can be handed, in one table: both
    /// caps AND the victim order.
    ///
    /// Catches, verified by mutation: dropping the `want_ladder` arm from
    /// `gdn_retention_caps` so `per_owner` is always the ladder width — the two
    /// no-ladder rows then read `(4, 4)`, `(5, 4)`. Separately, hardcoding
    /// `policy` to `Ladder` — the no-ladder rows then carry the ladder's
    /// victim order, which is the drift
    /// `a_persist_off_lineage_keeps_the_pre_ladder_victim_order` measures.
    #[test]
    fn retention_caps_gate_the_per_owner_bound_on_the_ladder_predicate() {
        let ladder = GDN_PREFIX_CHECKPOINTS_PER_OWNER;
        let pre_ladder = GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER;
        assert_ne!(
            ladder, pre_ladder,
            "the table below only says anything while the two caps differ"
        );

        assert_eq!(
            gdn_retention_caps(false, false),
            GdnRetentionCaps::pre_ladder(pre_ladder, pre_ladder)
        );
        assert_eq!(
            gdn_retention_caps(false, true),
            GdnRetentionCaps::pre_ladder(GDN_PREFIX_CHECKPOINT_LIMIT, pre_ladder)
        );
        assert_eq!(
            gdn_retention_caps(true, false),
            GdnRetentionCaps::ladder(ladder, ladder)
        );
        assert_eq!(
            gdn_retention_caps(true, true),
            GdnRetentionCaps::ladder(GDN_PREFIX_CHECKPOINT_LIMIT, ladder)
        );
    }

    /// One checkpoint at `prefix_len` on `tokens`, hashed the way the paged
    /// allocator hashes it, so `find_longest_valid_gdn_checkpoint_index` can be
    /// driven against the same token stream.
    fn owner_checkpoint(name: &'static str, tokens: &[u32], prefix_len: u32) -> Lineage {
        const BLOCK: u32 = 16;
        let extra_keys = vec![Vec::new(); tokens.len().div_ceil(BLOCK as usize)];
        let hashes = compute_paged_prefix_block_hashes(tokens, prefix_len, BLOCK, &extra_keys, 0)
            .expect("a block-aligned prefix must hash");
        Lineage {
            name,
            owner: "",
            prefix_len,
            block_size: BLOCK,
            final_block_hash: hashes.last().copied().unwrap_or_default(),
            tokens: tokens[..prefix_len as usize].to_vec(),
            hashes,
        }
    }

    /// The persistence-OFF defect the gate exists to stop, at the only layer
    /// where it is decidable without weights.
    ///
    /// Several conversations multiplexed over one model all publish under the
    /// same implicit owner id `""` (nothing but `mlx agent` ever sets one), and
    /// with no cold GDN policy each turn publishes exactly ONE rung. So the cap
    /// is not rationing a publisher's own ladder here — it decides whether the
    /// FIRST conversation's own entry is still there when its next turn asks.
    /// Losing it does not merely cost speed: the warm turn falls to
    /// `prepare_dense_gdn_prefix_state`'s `replay_materialized` arm, which
    /// rebuilds the recurrent state over a different span and emits different
    /// tokens (measured on real weights; see
    /// `GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER`).
    ///
    /// Catches, verified by mutation: ungating `gdn_retention_caps` so the
    /// no-ladder arm also gets the ladder width — the no-ladder half then
    /// retains 3 and HITS, i.e. persistence-OFF stops matching the pre-ladder
    /// behaviour.
    #[test]
    fn a_persist_off_owner_keeps_the_pre_ladder_number_of_sibling_lineages() {
        const BLOCK: u32 = 16;
        const NAMES: [&str; 3] = ["conv-a", "conv-b", "conv-c"];
        let conversations: Vec<Vec<u32>> = (0..3u32)
            .map(|conv| (0..2 * BLOCK).map(|index| conv * 1000 + index).collect())
            .collect();

        // Publish one deep rung per conversation, pruning after each push the
        // way `remember_*_gdn_materialized_prefix_checkpoint` does, then ask
        // whether the FIRST conversation can still be answered.
        let publish_all_then_look_up_the_first = |want_ladder: bool| -> (usize, Option<u32>) {
            let caps = gdn_retention_caps(want_ladder, false);
            let mut store: VecDeque<Lineage> = VecDeque::new();
            for (index, tokens) in conversations.iter().enumerate() {
                store.push_back(owner_checkpoint(NAMES[index], tokens, BLOCK));
                prune_gdn_checkpoints(&mut store, caps, "", "");
            }
            let extra_keys = vec![Vec::new(); 2];
            let hit = find_longest_valid_gdn_checkpoint_index(
                &store,
                "",
                &conversations[0],
                BLOCK,
                BLOCK,
                &extra_keys,
                0,
                |_| true,
            )
            .map(|index| store[index].prefix_len);
            (store.len(), hit)
        };

        assert_eq!(
            publish_all_then_look_up_the_first(false),
            (GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER, None),
            "a persistence-OFF turn must retain what it retained before the \
             ladder existed, so the third conversation still evicts the first"
        );
        assert_eq!(
            publish_all_then_look_up_the_first(true),
            (3, Some(BLOCK)),
            "a persist turn keeps the wider store the ladder needs, so the \
             first conversation is still answerable"
        );
    }

    /// The cap decides how many entries survive; the victim order decides WHICH.
    /// Both move the depth a later turn restores from, so both are gated.
    ///
    /// One monotone conversation publishes three rungs under one owner. Holding
    /// the caps fixed at the pre-ladder `(2, 2)` and varying ONLY the victim
    /// order, the two policies keep different pairs:
    ///
    /// ```text
    ///   published        16, 32, 48
    ///   PreLadder  ->    32, 48      (drops the first ancestor in publish order)
    ///   Ladder     ->    16, 48      (drops 32: smallest next/own ratio)
    /// ```
    ///
    /// A warm turn whose paged hit lands at 32 — a retry, an edited prompt, or a
    /// pool that evicted the tail — then restores from 32 under one and 16 under
    /// the other. Different replay span through
    /// `run_gdn_only_prefill_materialized`, different accumulation order,
    /// different tokens. 77e43031 shipped the `PreLadder` order, so a
    /// persistence-OFF turn has to keep taking it.
    ///
    /// Catches, verified by mutation:
    /// - pointing `PreLadder`'s two arms in `prune_gdn_checkpoints` at
    ///   `least_valuable_same_owner_ancestor` (i.e. never gating the order):
    ///   the `PreLadder` row below becomes `[16, 48]` / `Some(16)`.
    /// - hardcoding `gdn_retention_caps`'s `policy` to `Ladder`: the last
    ///   assertion, which drives the seam the two model call sites use rather
    ///   than naming a policy, fails the same way.
    #[test]
    fn a_persist_off_lineage_keeps_the_pre_ladder_victim_order() {
        const BLOCK: u32 = 16;
        let tokens: Vec<u32> = (0..4 * BLOCK).collect();
        let extra_keys = vec![Vec::new(); 4];

        let publish_three_rungs = |caps: GdnRetentionCaps| -> (Vec<u32>, Option<u32>) {
            let mut store: VecDeque<Lineage> = VecDeque::new();
            for rung in [BLOCK, 2 * BLOCK, 3 * BLOCK] {
                store.push_back(owner_checkpoint("conv", &tokens, rung));
                prune_gdn_checkpoints(&mut store, caps, "", "");
            }
            let retained: Vec<u32> = store.iter().map(|entry| entry.prefix_len).collect();
            // The warm turn asks for a prefix the deepest rung overshoots, so
            // the answer is whichever shallower rung survived.
            let hit = find_longest_valid_gdn_checkpoint_index(
                &store,
                "",
                &tokens,
                2 * BLOCK,
                BLOCK,
                &extra_keys,
                0,
                |_| true,
            )
            .map(|index| store[index].prefix_len);
            (retained, hit)
        };

        assert_eq!(
            publish_three_rungs(GdnRetentionCaps::pre_ladder(2, 2)),
            (vec![2 * BLOCK, 3 * BLOCK], Some(2 * BLOCK)),
            "the pre-ladder order drops the shallowest ancestor, so a monotone \
             conversation keeps its deepest rungs"
        );
        assert_eq!(
            publish_three_rungs(GdnRetentionCaps::ladder(2, 2)),
            (vec![BLOCK, 3 * BLOCK], Some(BLOCK)),
            "the ladder order keeps a shallow rung on purpose — only a turn that \
             writes a cold sidecar gets anything back for that"
        );

        // What the two model call sites actually hand `prune_gdn_checkpoints`
        // for a persistence-OFF turn. Named policies above prove the two orders
        // differ; this proves the seam picks the pre-ladder one.
        let persist_off = gdn_retention_caps(false, false);
        assert_eq!(persist_off.policy, GdnRetentionPolicy::PreLadder);
        assert_eq!(
            publish_three_rungs(persist_off),
            (vec![2 * BLOCK, 3 * BLOCK], Some(2 * BLOCK)),
            "a persistence-OFF turn must restore from the depth it restored \
             from before the ladder existed"
        );
    }

    #[test]
    fn linear_history_keeps_a_spread_of_boundaries_under_the_owner_cap() {
        let mut checkpoints: VecDeque<_> = (1..=10)
            .map(|len| {
                let hashes: Vec<u64> = (1..=len as u64).collect();
                lineage("linear", "root", &hashes)
            })
            .collect();

        prune_gdn_checkpoints(
            &mut checkpoints,
            GdnRetentionCaps::ladder(
                GDN_PREFIX_CHECKPOINT_LIMIT,
                GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            ),
            "root",
            "root",
        );

        let retained: Vec<usize> = checkpoints
            .iter()
            .map(|checkpoint| checkpoint.hashes.len())
            .collect();
        assert_eq!(
            retained.len(),
            GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            "{retained:?}"
        );
        assert!(
            retained.windows(2).all(|pair| pair[0] < pair[1]),
            "retained boundaries must stay ordered and distinct: {retained:?}"
        );
        // The head answers every warm lookup on this lineage.
        assert_eq!(retained[retained.len() - 1], 10, "{retained:?}");
        // A cold restore anchors only on the persisted part of the K/V chain,
        // which lags the prompt, so the shallow end has to survive the cap too.
        assert!(
            retained[0] <= 2,
            "the shallowest boundary was evicted: {retained:?}"
        );
    }

    #[test]
    fn root_and_four_active_child_owners_fit_without_eviction() {
        let mut checkpoints = VecDeque::from([lineage("parent", "root", &[1, 2])]);
        for branch in 0..4u64 {
            let owner = match branch {
                0 => "child-0",
                1 => "child-1",
                2 => "child-2",
                _ => "child-3",
            };
            checkpoints.push_back(lineage("child", owner, &[10 + branch]));
        }

        prune_gdn_checkpoints(
            &mut checkpoints,
            GdnRetentionCaps::ladder(
                GDN_PREFIX_CHECKPOINT_LIMIT,
                GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            ),
            "root",
            "child-3",
        );

        assert_eq!(checkpoints.len(), 5);
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "parent")
        );
    }

    #[test]
    fn sixth_owner_evicts_oldest_child_but_preserves_root() {
        let mut checkpoints = VecDeque::from([lineage("parent", "root", &[1, 2])]);
        for branch in 0..5u64 {
            let owner = match branch {
                0 => "child-0",
                1 => "child-1",
                2 => "child-2",
                3 => "child-3",
                _ => "child-4",
            };
            checkpoints.push_back(lineage("child", owner, &[100 + branch]));
        }

        prune_gdn_checkpoints(
            &mut checkpoints,
            GdnRetentionCaps::ladder(
                GDN_PREFIX_CHECKPOINT_LIMIT,
                GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            ),
            "root",
            "child-4",
        );

        assert_eq!(checkpoints.len(), 5);
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "parent")
        );
        assert!(
            !checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner == "child-0")
        );
    }

    #[test]
    fn rotating_root_evicts_old_root_and_preserves_new_root_with_four_children() {
        let mut checkpoints = VecDeque::from([lineage("old-root", "root-0", &[1])]);
        checkpoints.push_back(lineage("new-root", "root-1", &[2]));
        for (name, owner, hash) in [
            ("child-0", "child-0", 10),
            ("child-1", "child-1", 11),
            ("child-2", "child-2", 12),
            ("child-3", "child-3", 13),
        ] {
            checkpoints.push_back(lineage(name, owner, &[hash]));
            prune_gdn_checkpoints(
                &mut checkpoints,
                GdnRetentionCaps::ladder(
                    GDN_PREFIX_CHECKPOINT_LIMIT,
                    GDN_PREFIX_CHECKPOINTS_PER_OWNER,
                ),
                "root-1",
                owner,
            );
        }

        assert_eq!(checkpoints.len(), GDN_PREFIX_CHECKPOINT_LIMIT);
        assert!(
            !checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner == "root-0")
        );
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner == "root-1")
        );
    }

    #[test]
    fn equal_tokens_with_different_hash_chains_are_not_same_lineage() {
        let tokens = vec![1, 2];
        let mut checkpoints = VecDeque::from([
            Lineage {
                name: "salt-a",
                owner: "root",
                prefix_len: 2,
                block_size: 16,
                final_block_hash: 12,
                tokens: tokens.clone(),
                hashes: vec![11, 12],
            },
            Lineage {
                name: "salt-b",
                owner: "root",
                prefix_len: 2,
                block_size: 16,
                final_block_hash: 22,
                tokens,
                hashes: vec![21, 22],
            },
        ]);

        prune_gdn_checkpoints(
            &mut checkpoints,
            GdnRetentionCaps::ladder(1, 2),
            "root",
            "root",
        );

        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints.front().unwrap().name, "salt-b");
    }

    #[test]
    fn owner_cap_prefers_replacing_an_exact_ancestor() {
        let mut checkpoints = VecDeque::from([
            lineage("ancestor", "child", &[1]),
            lineage("other-branch", "child", &[9]),
            lineage("head", "child", &[1, 2]),
        ]);

        prune_gdn_checkpoints(
            &mut checkpoints,
            GdnRetentionCaps::ladder(5, 2),
            "root",
            "child",
        );

        assert_eq!(checkpoints.len(), 2);
        assert!(
            !checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "ancestor")
        );
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "other-branch")
        );
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "head")
        );
    }

    #[test]
    fn lineage_ancestry_rejects_different_block_sizes_and_owners() {
        let ancestor = lineage("ancestor", "root", &[1]);
        let mut other_block_size = lineage("descendant", "root", &[1, 2]);
        other_block_size.block_size = 32;
        assert!(!is_strict_ancestor(&ancestor, &other_block_size));

        let other_owner = lineage("descendant", "child", &[1, 2]);
        assert!(is_strict_ancestor(&ancestor, &other_owner));
        assert!(!is_same_owner_ancestor(&ancestor, &other_owner));
    }

    #[test]
    fn block_hash_chain_commits_salt_and_extra_keys() {
        let tokens: Vec<u32> = (1..=8).collect();
        let keys = vec![vec![11], vec![22]];
        let unsalted = compute_paged_prefix_block_hashes(&tokens, 8, 4, &keys, 0).unwrap();
        let salted = compute_paged_prefix_block_hashes(&tokens, 8, 4, &keys, 99).unwrap();
        let different_keys =
            compute_paged_prefix_block_hashes(&tokens, 8, 4, &[vec![12], vec![22]], 0).unwrap();

        assert_ne!(unsalted, salted);
        assert_ne!(unsalted, different_keys);
    }

    fn paged_lineage(
        name: &'static str,
        owner: &'static str,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        keys: &[Vec<u64>],
        salt: u64,
    ) -> Lineage {
        let hashes =
            compute_paged_prefix_block_hashes(tokens, prefix_len, block_size, keys, salt).unwrap();
        Lineage {
            name,
            owner,
            prefix_len,
            block_size,
            final_block_hash: *hashes.last().unwrap(),
            tokens: tokens[..prefix_len as usize].to_vec(),
            hashes,
        }
    }

    #[test]
    fn lookup_prefers_exact_then_longest_same_owner_ancestor() {
        let tokens: Vec<u32> = (1..=12).collect();
        let keys = vec![vec![11], vec![22], vec![33]];
        let mut checkpoints = VecDeque::from([
            paged_lineage("four", "root", &tokens, 4, 4, &keys, 7),
            paged_lineage("eight", "root", &tokens, 8, 4, &keys, 7),
            paged_lineage("other-owner", "child", &tokens, 12, 4, &keys, 7),
            paged_lineage("twelve", "root", &tokens, 12, 4, &keys, 7),
        ]);

        let exact = find_longest_valid_gdn_checkpoint_index(
            &checkpoints,
            "root",
            &tokens,
            12,
            4,
            &keys,
            7,
            |_| true,
        )
        .unwrap();
        assert_eq!(checkpoints[exact].name, "twelve");

        checkpoints.remove(exact);
        let ancestor = find_longest_valid_gdn_checkpoint_index(
            &checkpoints,
            "root",
            &tokens,
            12,
            4,
            &keys,
            7,
            |_| true,
        )
        .unwrap();
        assert_eq!(checkpoints[ancestor].name, "eight");

        checkpoints.remove(ancestor);
        checkpoints.push_back(paged_lineage(
            "descendant",
            "root",
            &tokens,
            12,
            4,
            &keys,
            7,
        ));
        let backed_up = find_longest_valid_gdn_checkpoint_index(
            &checkpoints,
            "root",
            &tokens,
            8,
            4,
            &keys,
            7,
            |_| true,
        )
        .unwrap();
        assert_eq!(checkpoints[backed_up].name, "four");
    }

    /// Drive the store the way a prefill drives it: publish every boundary the
    /// prefill planner asks for, pruning after each, then ask where a cold
    /// sidecar may be anchored under a K/V chain that only reaches
    /// `chain_blocks`.
    fn sidecar_boundary_after_prefill(
        prompt_tokens: u32,
        chain_blocks: u32,
        block_size: u32,
    ) -> GdnSidecarBoundary {
        let tokens: Vec<u32> = (0..prompt_tokens).collect();
        let keys = vec![Vec::new(); tokens.len().div_ceil(block_size as usize)];
        let mut checkpoints = VecDeque::new();
        for boundary in super::super::paged_forward::gdn_prefill_checkpoint_boundaries(
            tokens.len(),
            0,
            block_size,
        ) {
            checkpoints.push_back(paged_lineage(
                "prefill", "root", &tokens, boundary, block_size, &keys, 0,
            ));
            prune_gdn_checkpoints(
                &mut checkpoints,
                GdnRetentionCaps::ladder(
                    GDN_PREFIX_CHECKPOINT_LIMIT,
                    GDN_PREFIX_CHECKPOINTS_PER_OWNER,
                ),
                "root",
                "root",
            );
        }

        select_gdn_sidecar_boundary(
            &checkpoints,
            "root",
            &tokens,
            chain_blocks * block_size,
            block_size,
            &keys,
            0,
            |_| true,
        )
    }

    /// The persisted K/V chain advances only a few blocks per turn (the cold
    /// writer queue is bounded and `capture_chain` stops at the first block it
    /// refuses), so early turns on a long prompt see a chain reach far below the
    /// prompt's own end. A prefill must leave a checkpoint down there, otherwise
    /// no sidecar is ever written and the restore reuses zero blocks.
    #[test]
    fn cold_sidecar_anchors_under_a_partially_captured_chain() {
        let block_size = 16;
        let outcome = sidecar_boundary_after_prefill(4096, 24, block_size);

        let GdnSidecarBoundary::Selected { boundary, .. } = outcome else {
            panic!("no sidecar boundary under a 24-block chain reach: {outcome:?}");
        };
        let chain_reach = 24 * block_size;
        assert!(boundary > 0 && boundary <= chain_reach, "{boundary}");
        // Rungs are spaced by `GDN_CHECKPOINT_LADDER_RATIO`, so the rung under
        // any reachable chain covers more than 1/ratio of it. A ladder that
        // degenerated to one shallow rung would restore almost nothing.
        assert!(
            boundary * 4 > chain_reach,
            "rung {boundary} wastes most of a {chain_reach}-token chain"
        );
    }

    /// Drive several turns of one growing conversation through the store. Each
    /// turn prefills only the suffix past what the previous turn cached, so its
    /// ladder is short and the store accumulates rungs across turns until the
    /// owner cap bites.
    fn sidecar_boundary_after_turns(
        prompt_lens: &[u32],
        chain_blocks: u32,
        block_size: u32,
    ) -> GdnSidecarBoundary {
        let longest = prompt_lens.iter().copied().max().unwrap_or_default();
        let tokens: Vec<u32> = (0..longest).collect();
        let keys = vec![Vec::new(); tokens.len().div_ceil(block_size as usize)];
        let mut checkpoints = VecDeque::new();
        let mut cached_prefix_len = 0;
        for &prompt_len in prompt_lens {
            let prompt = &tokens[..prompt_len as usize];
            for boundary in super::super::paged_forward::gdn_prefill_checkpoint_boundaries(
                prompt.len(),
                cached_prefix_len,
                block_size,
            ) {
                checkpoints.push_back(paged_lineage(
                    "prefill", "root", prompt, boundary, block_size, &keys, 0,
                ));
                prune_gdn_checkpoints(
                    &mut checkpoints,
                    GdnRetentionCaps::ladder(
                        GDN_PREFIX_CHECKPOINT_LIMIT,
                        GDN_PREFIX_CHECKPOINTS_PER_OWNER,
                    ),
                    "root",
                    "root",
                );
            }
            cached_prefix_len = prompt_len;
        }

        select_gdn_sidecar_boundary(
            &checkpoints,
            "root",
            &tokens,
            chain_blocks * block_size,
            block_size,
            &keys,
            0,
            |_| true,
        )
    }

    /// A conversation that keeps growing publishes one deep rung per turn, so
    /// the owner cap starts evicting. The persisted chain advances at most a
    /// handful of blocks per turn, so it can still be far behind after several
    /// turns — evicting the shallow rungs first would leave that chain with
    /// nothing to anchor on.
    #[test]
    fn growing_conversation_keeps_a_rung_under_a_chain_that_still_lags() {
        let block_size = 16;
        let outcome =
            sidecar_boundary_after_turns(&[1024, 4096, 8192, 16384, 32768], 24, block_size);

        let GdnSidecarBoundary::Selected { boundary, .. } = outcome else {
            panic!("no sidecar boundary left after five turns: {outcome:?}");
        };
        let chain_reach = 24 * block_size;
        assert!(boundary > 0 && boundary <= chain_reach, "{boundary}");
        assert!(
            boundary * 4 > chain_reach,
            "rung {boundary} wastes most of a {chain_reach}-token chain"
        );
    }

    /// Highest number of GDN checkpoints alive at once across one prefill's
    /// publish.
    ///
    /// A prefill materializes every rung it crosses into a `Vec`
    /// (`run_paged_prefill_chunked`) and hands the whole `Vec` to
    /// `publish_dense_gdn_materialized_prefix_checkpoint_with_keys` only after
    /// the forward pass returns, so before the first push the entire ladder is
    /// resident alongside whatever the store already holds, and the undrained
    /// tail stays resident as the drain proceeds. `foreign_owners` seeds the
    /// store with other owners' checkpoints, the state a fresh turn can find it
    /// in.
    fn peak_resident_checkpoints(rungs: &[u32], foreign_owners: &[&'static str]) -> usize {
        let block_size = 16u32;
        let foreign_span = block_size * foreign_owners.len() as u32;
        let longest = rungs
            .iter()
            .copied()
            .max()
            .unwrap_or_default()
            .max(foreign_span);
        let tokens: Vec<u32> = (0..longest).collect();
        let keys = vec![Vec::new(); tokens.len().div_ceil(block_size as usize)];

        let mut checkpoints = VecDeque::new();
        for (idx, owner) in foreign_owners.iter().enumerate() {
            checkpoints.push_back(paged_lineage(
                "other-turn",
                owner,
                &tokens,
                block_size * (idx as u32 + 1),
                block_size,
                &keys,
                0,
            ));
            prune_gdn_checkpoints(
                &mut checkpoints,
                GdnRetentionCaps::ladder(
                    GDN_PREFIX_CHECKPOINT_LIMIT,
                    GDN_PREFIX_CHECKPOINTS_PER_OWNER,
                ),
                "root",
                owner,
            );
        }

        // Every rung is materialized before any of them is published.
        let mut peak = checkpoints.len() + rungs.len();
        for (idx, &boundary) in rungs.iter().enumerate() {
            checkpoints.push_back(paged_lineage(
                "prefill", "root", &tokens, boundary, block_size, &keys, 0,
            ));
            prune_gdn_checkpoints(
                &mut checkpoints,
                GdnRetentionCaps::ladder(
                    GDN_PREFIX_CHECKPOINT_LIMIT,
                    GDN_PREFIX_CHECKPOINTS_PER_OWNER,
                ),
                "root",
                "root",
            );
            peak = peak.max(checkpoints.len() + (rungs.len() - idx - 1));
        }
        peak
    }

    /// `GDN_PREFIX_CHECKPOINT_LIMIT` bounds what the store RETAINS, which is not
    /// the high-water mark: the rungs a prefill has materialized but not yet
    /// handed over are live at the same time. This pins the real peak, which is
    /// the number the memory figure beside `GDN_CHECKPOINT_LADDER_RUNGS` is
    /// computed from.
    #[test]
    fn publishing_a_ladder_peaks_at_the_store_bound_plus_the_whole_ladder() {
        let ladder = super::super::paged_forward::gdn_prefill_checkpoint_boundaries(4096, 0, 16);
        assert_eq!(ladder.len(), 4, "{ladder:?}");

        // An idle store: the ladder itself is the whole peak.
        assert_eq!(peak_resident_checkpoints(&ladder, &[]), ladder.len());

        // Worst case the caps admit: the store is already full of other owners'
        // checkpoints when this turn publishes. Pruning cannot help before the
        // first push, because the ladder is not in the store yet.
        let others = ["a", "b", "c", "d", "e"];
        assert_eq!(others.len(), GDN_PREFIX_CHECKPOINT_LIMIT);
        assert_eq!(
            peak_resident_checkpoints(&ladder, &others),
            GDN_PREFIX_CHECKPOINT_LIMIT + ladder.len()
        );
    }

    const LADDER_PROMPT_TOKENS: u32 = 4096;
    const LADDER_BLOCK_SIZE: u32 = 16;
    /// `gdn_prefill_checkpoint_boundaries(4096, 0, 16)`, pinned by
    /// `paged_forward::tests::ladder_rungs_are_quarters_of_the_one_above`.
    const FULL_LADDER: [u32; 4] = [48, 240, 1008, 4080];

    fn ladder_tokens_and_keys() -> (Vec<u32>, Vec<Vec<u64>>) {
        let tokens: Vec<u32> = (0..LADDER_PROMPT_TOKENS).collect();
        let keys = vec![Vec::new(); tokens.len().div_ceil(LADDER_BLOCK_SIZE as usize)];
        (tokens, keys)
    }

    /// Push one turn's whole ladder the way production pushes it: rung by rung,
    /// ascending, with a prune after EVERY rung (`prune` is called from
    /// `remember_*_gdn_materialized_prefix_checkpoint`, which
    /// `publish_*_with_keys` calls once per rung).
    fn publish_ladder(
        checkpoints: &mut VecDeque<Lineage>,
        active: &'static str,
        root: &'static str,
        limit: usize,
    ) {
        let (tokens, keys) = ladder_tokens_and_keys();
        let ladder = super::super::paged_forward::gdn_prefill_checkpoint_boundaries(
            tokens.len(),
            0,
            LADDER_BLOCK_SIZE,
        );
        assert_eq!(ladder, FULL_LADDER, "ladder shape moved");
        for boundary in ladder {
            checkpoints.push_back(paged_lineage(
                "prefill",
                active,
                &tokens,
                boundary,
                LADDER_BLOCK_SIZE,
                &keys,
                0,
            ));
            prune_gdn_checkpoints(
                checkpoints,
                GdnRetentionCaps::ladder(limit, GDN_PREFIX_CHECKPOINTS_PER_OWNER),
                root,
                active,
            );
        }
    }

    /// Seed the store with what other owners already left in it, as
    /// `(owner, boundaries)` pushed in order. This is the state a live process
    /// hands a fresh turn: sibling subagent branches, or the residue of a
    /// session id the `/new` rotation retired.
    fn seed_foreign_owners(
        checkpoints: &mut VecDeque<Lineage>,
        resident: &[(&'static str, &'static [u32])],
        root: &'static str,
        limit: usize,
    ) {
        let (tokens, keys) = ladder_tokens_and_keys();
        for (owner, boundaries) in resident {
            for &boundary in *boundaries {
                checkpoints.push_back(paged_lineage(
                    "resident",
                    owner,
                    &tokens,
                    boundary,
                    LADDER_BLOCK_SIZE,
                    &keys,
                    0,
                ));
                prune_gdn_checkpoints(
                    checkpoints,
                    GdnRetentionCaps::ladder(limit, GDN_PREFIX_CHECKPOINTS_PER_OWNER),
                    root,
                    owner,
                );
            }
        }
    }

    fn retained_boundaries(checkpoints: &VecDeque<Lineage>, owner: &str) -> Vec<u32> {
        checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.owner == owner)
            .map(|checkpoint| checkpoint.prefix_len)
            .collect()
    }

    fn store_shape(checkpoints: &VecDeque<Lineage>) -> Vec<(&str, u32)> {
        checkpoints
            .iter()
            .map(|checkpoint| (checkpoint.owner, checkpoint.prefix_len))
            .collect()
    }

    fn sidecar_boundary_for(
        checkpoints: &VecDeque<Lineage>,
        owner: &str,
        chain_blocks: u32,
    ) -> GdnSidecarBoundary {
        let (tokens, keys) = ladder_tokens_and_keys();
        select_gdn_sidecar_boundary(
            checkpoints,
            owner,
            &tokens,
            chain_blocks * LADDER_BLOCK_SIZE,
            LADDER_BLOCK_SIZE,
            &keys,
            0,
            |_| true,
        )
    }

    /// The exact shape `GDN_PREFIX_CHECKPOINT_LIMIT = 5` was sized for: a root
    /// session plus four concurrent subagent branches, each holding one
    /// checkpoint, with the root now publishing a turn's ladder.
    ///
    /// This is the case where preferring a foreign victim buys the publisher
    /// nothing, because there IS no foreign victim to prefer: four distinct
    /// owners with one entry each hold no redundancy anywhere. Arm 1 finds
    /// nothing, arm 2 takes the publisher's own newest-redundant rung after
    /// every push, and the ladder eats itself down to the single endpoint rung
    /// it exists to replace — the capture at the end of this same turn misses.
    ///
    /// What that buys instead is every sibling's warm reuse: nobody is pushed to
    /// zero checkpoints. That is the trade, priced honestly. A missed sidecar
    /// costs one SSD write only a restart would have read; a blinded sibling
    /// re-forwards its whole cached prefix on its next turn and, by (F1) and
    /// (F2), never republishes a shallow rung to recover with.
    #[test]
    fn four_single_entry_siblings_outlive_the_publishers_ladder() {
        let resident: [(&'static str, &'static [u32]); 4] = [
            ("child-0", &[16]),
            ("child-1", &[32]),
            ("child-2", &[64]),
            ("child-3", &[80]),
        ];
        let mut store = VecDeque::new();
        seed_foreign_owners(&mut store, &resident, "root", GDN_PREFIX_CHECKPOINT_LIMIT);
        assert_eq!(store.len(), 4, "{:?}", store_shape(&store));

        publish_ladder(&mut store, "root", "root", GDN_PREFIX_CHECKPOINT_LIMIT);

        assert_eq!(
            store_shape(&store),
            vec![
                ("child-0", 16),
                ("child-1", 32),
                ("child-2", 64),
                ("child-3", 80),
                ("root", 4080),
            ],
            "every sibling must keep the one checkpoint it holds, and the \
             publisher pays the coverage instead"
        );
        // A chain 24 blocks deep is far below the prompt's own end, which is
        // where every early turn sits. The ladder's job is to have a rung down
        // there; the collapsed ladder has none, so this turn writes no sidecar.
        assert_eq!(
            sidecar_boundary_for(&store, "root", 24),
            GdnSidecarBoundary::Missed(GdnSidecarBoundaryMiss {
                deepest_retained: Some(4080),
                retained_for_owner: 1,
                retained_total: GDN_PREFIX_CHECKPOINT_LIMIT,
            }),
            "{:?}",
            store_shape(&store)
        );
    }

    /// The same shape without an explicit root, where the global cap is
    /// `GDN_PREFIX_CHECKPOINTS_PER_OWNER` instead. Reachable through
    /// `buildChatConfig`, which sets `cacheOwnerId` from the session id but
    /// leaves `cacheRootOwnerId` unset until the session manager has published
    /// a root id. The retention policy must not depend on which limit is in
    /// force.
    ///
    /// Five live owners in four slots is past the cliff, so exactly one owner
    /// does lose its last checkpoint — the oldest, on the first push, before any
    /// redundancy exists to take instead. Every owner after that is kept.
    #[test]
    fn siblings_outlive_the_publishers_ladder_at_the_implicit_limit() {
        let resident: [(&'static str, &'static [u32]); 4] = [
            ("child-0", &[16]),
            ("child-1", &[32]),
            ("child-2", &[64]),
            ("child-3", &[80]),
        ];
        let mut store = VecDeque::new();
        seed_foreign_owners(
            &mut store,
            &resident,
            "root",
            GDN_PREFIX_CHECKPOINTS_PER_OWNER,
        );
        assert_eq!(store.len(), 4, "{:?}", store_shape(&store));

        publish_ladder(&mut store, "root", "root", GDN_PREFIX_CHECKPOINTS_PER_OWNER);

        assert_eq!(
            store_shape(&store),
            vec![
                ("child-1", 32),
                ("child-2", 64),
                ("child-3", 80),
                ("root", 4080),
            ],
            "one owner over the cap costs exactly one owner's last checkpoint"
        );
    }

    /// A sibling branch that holds a whole ladder of its own. Here foreign
    /// ancestors DO exist, so "drop the most redundant rung" has somewhere else
    /// to look — but unrestricted it interleaves, taking the publisher's fresh
    /// rungs as soon as they become the most redundant pair in the store.
    #[test]
    fn a_ladder_outlives_a_foreign_owner_holding_its_own_ladder() {
        let resident: [(&'static str, &'static [u32]); 1] = [("child-0", &FULL_LADDER)];
        let mut store = VecDeque::new();
        seed_foreign_owners(&mut store, &resident, "root", GDN_PREFIX_CHECKPOINT_LIMIT);
        assert_eq!(retained_boundaries(&store, "child-0"), FULL_LADDER.to_vec());

        publish_ladder(&mut store, "root", "root", GDN_PREFIX_CHECKPOINT_LIMIT);

        assert_eq!(
            retained_boundaries(&store, "root"),
            FULL_LADDER.to_vec(),
            "the publishing owner lost rungs: {:?}",
            store_shape(&store)
        );
        // The sibling pays for it, and pays in the right order: its ladder is
        // eaten from the most redundant rung down, so what it keeps is the
        // deepest one, which still answers its next warm turn.
        assert_eq!(
            retained_boundaries(&store, "child-0"),
            vec![4080],
            "{:?}",
            store_shape(&store)
        );
    }

    /// `/new` and `/resume` rotate the root session id and every pi subagent
    /// mints its own, so one model instance sees owner ids alternate turn after
    /// turn. While the retired owners still hold redundancy, arm 1 spends theirs
    /// and whoever publishes keeps its SHALLOWEST rung — the only one a cold
    /// capture can anchor on while the persisted chain still lags.
    ///
    /// The store shrinks each owner to its deepest rung as the rotation goes on,
    /// so the publisher's own ladder pays more each round: the whole ladder,
    /// then three rungs, then two. What never happens is an owner going to zero.
    #[test]
    fn each_owner_in_a_rotation_keeps_a_shallow_rung_of_what_it_published() {
        let mut store = VecDeque::new();
        seed_foreign_owners(
            &mut store,
            &[("root", &[16])],
            "root",
            GDN_PREFIX_CHECKPOINT_LIMIT,
        );

        let mut published: Vec<&str> = Vec::new();
        for owner in ["session-a", "session-b", "session-c"] {
            publish_ladder(&mut store, owner, "root", GDN_PREFIX_CHECKPOINT_LIMIT);
            published.push(owner);

            let retained = retained_boundaries(&store, owner);
            assert_eq!(
                retained.first().copied(),
                Some(FULL_LADDER[0]),
                "{owner} lost the shallow rung it had just published, so its \
                 capture this turn has nothing under the chain's reach: {:?}",
                store_shape(&store)
            );
            assert_eq!(
                retained.last().copied(),
                Some(FULL_LADDER[FULL_LADDER.len() - 1]),
                "{owner} lost its deepest rung: {:?}",
                store_shape(&store)
            );
            assert_eq!(
                retained_boundaries(&store, "root"),
                vec![16],
                "the interactive parent was displaced by a task owner: {:?}",
                store_shape(&store)
            );
            for earlier in &published {
                assert!(
                    !retained_boundaries(&store, earlier).is_empty(),
                    "{earlier} was pushed to zero checkpoints and by (F1) will \
                     never republish: {:?}",
                    store_shape(&store)
                );
            }
        }
    }

    /// A subagent publishing while its siblings and the root are all resident,
    /// each holding one checkpoint. Nobody else has redundancy to give, so the
    /// publisher's ladder collapses to its deepest rung and every other owner —
    /// root included — keeps exactly what it had.
    #[test]
    fn a_subagent_turn_leaves_every_sibling_and_the_root_a_checkpoint() {
        let resident: [(&'static str, &'static [u32]); 4] = [
            ("root", &[16]),
            ("child-0", &[32]),
            ("child-1", &[64]),
            ("child-2", &[80]),
        ];
        let mut store = VecDeque::new();
        seed_foreign_owners(&mut store, &resident, "root", GDN_PREFIX_CHECKPOINT_LIMIT);

        publish_ladder(&mut store, "child-3", "root", GDN_PREFIX_CHECKPOINT_LIMIT);

        assert_eq!(
            store_shape(&store),
            vec![
                ("root", 16),
                ("child-0", 32),
                ("child-1", 64),
                ("child-2", 80),
                ("child-3", 4080),
            ],
            "a subagent turn must not blind the root or a sibling"
        );
    }

    /// What protecting the publisher costs the interactive parent. This is NOT
    /// desired behaviour — it is the measured price of keeping a 4-rung ladder
    /// in a 5-slot store, pinned so it cannot change unnoticed.
    ///
    /// The root's own ladder is the PREFERRED victim, because
    /// `least_valuable_same_owner_ancestor` runs first and has no root guard: a
    /// ladder is nothing but same-owner ancestors, so it looks maximally
    /// redundant. One subagent turn eats it down to its deepest rung, and a warm
    /// turn republishes only its deepest rung (module doc), so the root's
    /// shallow coverage never returns and every later capture of the parent
    /// session misses while its persisted chain still lags.
    #[test]
    fn one_subagent_turn_strips_the_root_to_its_deepest_rung() {
        let resident: [(&'static str, &'static [u32]); 1] = [("root", &FULL_LADDER)];
        let mut store = VecDeque::new();
        seed_foreign_owners(&mut store, &resident, "root", GDN_PREFIX_CHECKPOINT_LIMIT);
        assert_eq!(retained_boundaries(&store, "root"), FULL_LADDER.to_vec());
        // Before the child's turn the parent can anchor a capture.
        assert!(
            matches!(
                sidecar_boundary_for(&store, "root", 24),
                GdnSidecarBoundary::Selected { boundary: 240, .. }
            ),
            "{:?}",
            store_shape(&store)
        );

        publish_ladder(&mut store, "child-0", "root", GDN_PREFIX_CHECKPOINT_LIMIT);

        assert_eq!(
            retained_boundaries(&store, "child-0"),
            FULL_LADDER.to_vec(),
            "{:?}",
            store_shape(&store)
        );
        assert_eq!(
            retained_boundaries(&store, "root"),
            vec![4080],
            "the root kept more than its deepest rung; re-measure the cost \
             this test records: {:?}",
            store_shape(&store)
        );
        assert_eq!(
            sidecar_boundary_for(&store, "root", 24),
            GdnSidecarBoundary::Missed(GdnSidecarBoundaryMiss {
                deepest_retained: Some(4080),
                retained_for_owner: 1,
                retained_total: GDN_PREFIX_CHECKPOINT_LIMIT,
            }),
            "{:?}",
            store_shape(&store)
        );
    }

    /// Four live owners in five slots, one of them holding a spare rung. This is
    /// the case both arms are visible in at once, so it is the one that pins the
    /// order they run in.
    ///
    /// Arm 1 spends the root's spare rung — the root's LAST entry is protected,
    /// its coverage is not — which is what leaves the publisher holding a
    /// shallow rung as well as its deepest, so its capture this turn can anchor.
    /// Arm 3, the only eviction that blinds an owner, is never reached: with
    /// four owners in five slots there is always a redundant entry somewhere.
    #[test]
    fn one_subagent_turn_leaves_every_sibling_a_checkpoint() {
        let resident: [(&'static str, &'static [u32]); 3] = [
            ("root", &[48, 4080]),
            ("child-0", &[4080]),
            ("child-1", &[4080]),
        ];
        let mut store = VecDeque::new();
        seed_foreign_owners(&mut store, &resident, "root", GDN_PREFIX_CHECKPOINT_LIMIT);
        assert_eq!(store.len(), 4, "{:?}", store_shape(&store));

        publish_ladder(&mut store, "child-2", "root", GDN_PREFIX_CHECKPOINT_LIMIT);

        assert_eq!(
            store_shape(&store),
            vec![
                ("root", 4080),
                ("child-0", 4080),
                ("child-1", 4080),
                ("child-2", 48),
                ("child-2", 4080),
            ],
            "the publisher keeps a shallow rung because a sibling had a spare \
             one to give, and nobody is pushed to zero"
        );
        // The shallow rung is not decoration: it is the whole reason this turn
        // can persist anything while its chain is still 24 blocks in.
        assert!(
            matches!(
                sidecar_boundary_for(&store, "child-2", 24),
                GdnSidecarBoundary::Selected { boundary: 48, .. }
            ),
            "{:?}",
            store_shape(&store)
        );
    }

    /// Same store, a chain that has ratcheted all the way to the prompt's own
    /// checkpoint boundary. This is the case that already works today and must
    /// keep picking the deepest boundary.
    #[test]
    fn cold_sidecar_anchors_at_the_prompt_boundary_once_the_chain_reaches_it() {
        let block_size = 16;
        let outcome = sidecar_boundary_after_prefill(4096, 4096 / 16, block_size);

        let GdnSidecarBoundary::Selected { boundary, .. } = outcome else {
            panic!("no sidecar boundary under a fully captured chain: {outcome:?}");
        };
        assert_eq!(boundary, 4080);
    }

    #[test]
    fn lookup_rejects_token_salt_extra_key_and_cache_readiness_mismatches() {
        let tokens: Vec<u32> = (1..=8).collect();
        let keys = vec![vec![11], vec![22]];
        let checkpoint = paged_lineage("candidate", "root", &tokens, 8, 4, &keys, 7);
        let checkpoints = VecDeque::from([checkpoint]);

        let mut divergent_tokens = tokens.clone();
        divergent_tokens[6] = 99;
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &divergent_tokens,
                8,
                4,
                &keys,
                7,
                |_| true,
            )
            .is_none()
        );
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &tokens,
                8,
                4,
                &keys,
                8,
                |_| true,
            )
            .is_none()
        );
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &tokens,
                8,
                4,
                &[vec![11], vec![23]],
                7,
                |_| true,
            )
            .is_none()
        );
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &tokens,
                8,
                4,
                &keys,
                7,
                |_| false,
            )
            .is_none()
        );
    }
}
