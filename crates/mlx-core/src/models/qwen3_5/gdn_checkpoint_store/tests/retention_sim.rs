//! Multi-owner, multi-turn measurement of what the bounded GDN checkpoint store
//! delivers to its two consumers.
//!
//! The tests beside this one are characterizations: one store, one publish, one
//! assertion. They cannot answer the only question that decides the eviction
//! policy — over a whole conversation, with several owners interleaved, how many
//! cold sidecars get written and how many prefix tokens get re-forwarded through
//! the GDN layers. This module answers both by running turns.
//!
//! ```text
//!            per owner-turn
//!   prompt ──────────────────────────────────────────────────────────┐
//!     │                                                              │
//!     ├─(1) cached_prefix_len  = block-aligned reuse of the last turn │
//!     │                                                              │
//!     ├─(2) WARM  find_longest_valid_gdn_checkpoint_index  ───────────┤ metric B
//!     │        hit  L  -> replay cached-L        miss -> replay cached│ (hot path)
//!     │                                                              │
//!     ├─(3) publish gdn_prefill_checkpoint_boundaries, rung by rung,  │
//!     │        prune_gdn_checkpoints after EVERY rung                 │
//!     │                                                              │
//!     └─(4) COLD  select_gdn_sidecar_boundary(chain reach) ───────────┘ metric A
//!              new boundary -> sidecar written    seen before -> dedup
//! ```
//!
//! Steps 2, 3 and 4 call the shipped functions. Only the two things that need a
//! GPU or a disk are modelled: the forward pass (a checkpoint is just its
//! identity here) and the SSD writer's ratchet, whose shape is taken from
//! [`ColdTierWalk::capture_chain`] — see [`QUEUE_BLOCKS_PER_TURN`].
//!
//! The two global-cap arms run from one binary. The shipped arm passes the
//! publishing owner as `active_owner_id`; the control arm passes an id no
//! checkpoint carries, which makes `OwnerScope::Excluding(active)` admit exactly
//! what `OwnerScope::Any` admits and collapses `prune_gdn_checkpoints` onto the
//! unrestricted search it had before. `an_absent_active_owner_is_the_unrestricted_arm`
//! and `the_two_arms_disagree_about_who_pays` pin that equivalence.
//!
//! What forty agent turns at the shipped caps say, per owner count. Round-robin
//! first, then burst x4 — the shape pi actually produces, four task loops
//! interleaved by the host:
//!
//! ```text
//!   owners                     1     2     3     4     5     6
//!   round-robin
//!     sidecars   Excluding    38    38    35    32    24    20
//!                Any          38    35    32    27    22    18
//!     replay %   BOTH        0.0   5.9   7.8   9.3  10.5  84.0
//!     blind      BOTH          0     0     0     0     0    28
//!   burst x4
//!     sidecars   Excluding    38    36    33    30    26    22
//!                Any          38    35    31    27    23    19
//!     replay %   BOTH        0.0   1.2   1.5   1.5   1.5   9.6
//!     blind      BOTH          0     0     0     0     0     3
//!                                                          ^^^^
//!                                              capacity, both arms
//! ```
//!
//! The sidecar rows are at the shipped chain speed, [`QUEUE_BLOCKS_PER_TURN`].
//! They read 2-7 instead of 20-38 under the pre-`fsync(2)` writer, which
//! persisted 8-9 blocks a turn against this fixture's 16 — a chain that never
//! advanced past the first few boundaries. The replay and blind rows are the
//! same at both speeds.
//!
//! The metric-B rows are not rounded to look alike: `gdn_replay_tokens` and the
//! blind-turn count are EQUAL between the arms, in all 324 cells of
//! `the_publisher_arm_never_costs_hot_path_replay`'s sweep, for the structural
//! reason set out in `prune_gdn_checkpoints`'s doc. The victim ORDER moves metric
//! A only. What moves metric B is capacity: six owners in five slots collapses
//! under either arm, and every `CAPACITY_SWEEP` variant at limit 6 or above
//! reads 0.0 / 5.9 / 7.8 / 9.3 / 10.5 / 11.4 instead — for +75 MiB of resident
//! 27B checkpoints per slot.

use std::collections::BTreeMap;
use std::collections::HashSet;

use super::*;
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::gdn_sidecar;
use crate::models::qwen3_5::paged_forward::gdn_prefill_checkpoint_boundaries;

/// Blocks the persisted K/V chain advances per turn.
///
/// `ColdTierWalk::capture_chain` walks a request's blocks from index 0, skips
/// every block `contains` already reports on disk at no cost, and spends an
/// explicit per-turn budget — `MLX_COLD_CAPTURE_BLOCKS_PER_TURN`, default 128
/// — on the rest, waiting for a writer-queue slot rather than stopping when it
/// finds none. So the ratchet is this number, exactly, on any filesystem.
///
/// It used to be an emergent quantity instead: the walk broke at the first
/// block the bounded queue refused, which put the frontier at
/// `N = (Q + 1) / (1 - Tc/Tw)` — 25-28 dense and 17-18 MoE off
/// `cold_cache::bench_chain_advance_per_turn`, 8-9 while the writer still
/// called `F_FULLFSYNC` per object, and unbounded on a filesystem faster than
/// the capture. `docs/paged-cache.md` carries the whole derivation as history.
///
/// This is the one number here that is a dial rather than a call, so
/// `chain_speed_changes_the_level_not_the_ranking` re-runs the sweep at half
/// and double this value. The pre-budget rates are swept alongside it
/// ([`PRE_FSYNC_BLOCKS_PER_TURN`]), because the retention ranking has to hold
/// at a chain speed an order of magnitude slower than the shipped one.
const QUEUE_BLOCKS_PER_TURN: u32 = 128;

/// The ratchet before the writer stopped calling `F_FULLFSYNC` per object, and
/// before the walk had a budget at all.
///
/// Kept as a sweep point because it is the slowest chain speed the retention
/// ranking is ever asked to hold at: a cold tier on a filesystem where
/// `fsync(2)` is as expensive as `F_FULLFSYNC` still spends its whole budget
/// waiting, and lands here.
const PRE_FSYNC_BLOCKS_PER_TURN: u32 = 8;

/// Owner ids the sweep hands out, in the order it hands them out. `OWNERS[0]` is
/// the root — the interactive session that `mlx agent` names explicitly and that
/// `prune_gdn_checkpoints` protects one entry of.
const OWNERS: [&str; 6] = [
    "owner-0", "owner-1", "owner-2", "owner-3", "owner-4", "owner-5",
];

/// An `active_owner_id` no checkpoint in the store can carry, so
/// `OwnerScope::Excluding(ABSENT_ACTIVE_OWNER)` admits every stored entry.
const ABSENT_ACTIVE_OWNER: &str = "\u{0}absent-active-owner";

/// Which global-cap arm a run exercises.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Policy {
    /// Shipped: the publishing owner's ladder is the last thing evicted.
    ProtectsPublisher,
    /// The unrestricted search `prune_gdn_checkpoints` used before, reached by
    /// naming an owner the store does not hold.
    Unrestricted,
}

impl Policy {
    fn active_owner(self, publishing: &str) -> &str {
        match self {
            Policy::ProtectsPublisher => publishing,
            Policy::Unrestricted => ABSENT_ACTIVE_OWNER,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Policy::ProtectsPublisher => "APPLIED  Excluding(active)",
            Policy::Unrestricted => "PRE-FIX  Any             ",
        }
    }
}

/// The two bounds `prune_gdn_checkpoints` enforces, so a run can sweep capacity
/// without touching the shipped constants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Caps {
    limit: usize,
    per_owner: usize,
}

impl Caps {
    const SHIPPED: Self = Self {
        limit: GDN_PREFIX_CHECKPOINT_LIMIT,
        per_owner: GDN_PREFIX_CHECKPOINTS_PER_OWNER,
    };
}

/// Shape of one conversation: how long the first prompt is and how much every
/// later turn adds.
#[derive(Clone, Copy)]
struct Fixture {
    label: &'static str,
    block_size: u32,
    first_prompt_tokens: u32,
    user_tokens: u32,
    assistant_tokens: u32,
}

impl Fixture {
    /// An agentic session: a large seeded context and turns that carry tool
    /// output. It opens 64 blocks (1024 tokens) behind and each of an owner's
    /// own turns adds 16 more.
    ///
    /// At the shipped [`QUEUE_BLOCKS_PER_TURN`] the chain clears all 64 on the
    /// turn that opens them and never falls behind again, which is what the
    /// capture budget was added to do — so this fixture no longer exercises the
    /// behind-the-prompt regime at the shipped speed, and the sweep in
    /// `chain_speed_changes_the_level_not_the_ranking` reaches that regime
    /// through its slower points instead. At 16 blocks/turn the chain gains
    /// nothing per own-turn and the gap persists; under the pre-`fsync(2)`
    /// writer ([`PRE_FSYNC_BLOCKS_PER_TURN`]) it LOST 8 blocks per own-turn and
    /// never caught up at all. That span is the regime the checkpoint ladder
    /// was added for.
    const AGENT: Self = Self {
        label: "agent   1024+256/turn",
        block_size: 16,
        first_prompt_tokens: 1024,
        user_tokens: 64,
        assistant_tokens: 192,
    };

    /// Short back-and-forth chat: turns add 6 blocks against a chain that
    /// ratchets [`QUEUE_BLOCKS_PER_TURN`], so the persisted chain overtakes the
    /// prompt within two turns and every later turn can anchor at its own
    /// deepest rung. At the shipped chain speed that saturates — see
    /// `the_publisher_arm_buys_cold_sidecars`.
    const CHAT: Self = Self {
        label: "chat     256+96/turn",
        block_size: 16,
        first_prompt_tokens: 256,
        user_tokens: 32,
        assistant_tokens: 64,
    };

    /// The shape the standalone Python simulator swept, so its table and this
    /// one can be compared row for row. It charges the whole turn to the prompt
    /// and generates nothing, which makes each turn's cached prefix exactly the
    /// previous prompt.
    const PRIOR_PASS: Self = Self {
        label: "prior   4096+2048/turn",
        block_size: 16,
        first_prompt_tokens: 4096,
        user_tokens: 2048,
        assistant_tokens: 0,
    };

    fn turn_tokens(&self) -> u32 {
        self.user_tokens + self.assistant_tokens
    }

    /// Prompt length on this owner's `turn`-th own turn (0-based).
    fn prompt_len(&self, turn: usize) -> u32 {
        self.first_prompt_tokens + self.turn_tokens() * turn as u32
    }

    /// Tokens the request holds once the turn has decoded, which is what the
    /// end-of-turn K/V finalize and the cold capture both see.
    fn total_len(&self, turn: usize) -> u32 {
        self.prompt_len(turn) + self.assistant_tokens
    }
}

fn align_down(tokens: u32, block_size: u32) -> u32 {
    tokens / block_size * block_size
}

/// Everything one run reports. Counters are per run, not per owner.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Metrics {
    turns: usize,
    /// Metric A: turns that actually enqueued a sidecar.
    cold_captures: usize,
    /// Turns where a boundary was selected but that exact chain was already on
    /// disk (`contains_in` dedup), so nothing was written.
    cold_dedup_skips: usize,
    /// Turns where the persisted chain covered no whole block yet.
    cold_chain_empty: usize,
    /// Turns where the chain had reach but no retained rung sat under it.
    cold_boundary_misses: usize,
    /// Metric B numerator: prefix tokens re-forwarded through the GDN layers.
    gdn_replay_tokens: u64,
    /// Metric B denominator: every cached prefix token a turn started from.
    cached_prefix_tokens: u64,
    /// Turns whose GDN state came from the still-live caches (same owner ran
    /// last), so the store was never consulted.
    warm_live: usize,
    /// Turns served by the single-slot history checkpoint.
    warm_history_slot: usize,
    /// Turns the store answered exactly (zero replay).
    warm_store_exact: usize,
    /// Turns the store answered with a shallower rung (partial replay).
    warm_store_partial: usize,
    /// Turns with no usable checkpoint at all: the whole cached prefix is
    /// re-forwarded and `cached_prefix_len` is NOT reset, so this repeats.
    warm_full_replay: usize,
    /// Sum over owners of the DEEPEST boundary that owner ever got onto disk.
    /// A sidecar's cost does not depend on its boundary, so this — not the
    /// write count — is what a restart gets back for the bytes.
    restore_depth_tokens: u64,
    /// The shallowest of those per-owner depths, i.e. the worst-served owner.
    restore_depth_min: u32,
}

impl Metrics {
    fn replay_fraction(&self) -> f64 {
        if self.cached_prefix_tokens == 0 {
            return 0.0;
        }
        self.gdn_replay_tokens as f64 / self.cached_prefix_tokens as f64
    }

    /// Turns where a rung was found under the chain's reach, whether or not that
    /// chain was already on disk. This is what a capture counter without the
    /// `contains_in` dedup counts, and it is the column the standalone Python
    /// simulator reported as "hits".
    fn cold_selectable(&self) -> usize {
        self.cold_captures + self.cold_dedup_skips
    }
}

/// The order the shared model sees turns in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Order {
    /// One turn each, cycling. Every owner is idle for exactly `owners - 1`
    /// turns between its own.
    RoundRobin,
    /// What the subagent extension actually produces: a task loop takes the
    /// model for several consecutive turns while every other owner waits.
    Burst(usize),
}

impl Order {
    fn owner_at(self, turn: usize, owners: usize) -> usize {
        match self {
            Order::RoundRobin => turn % owners,
            Order::Burst(run_len) => (turn / run_len.max(1)) % owners,
        }
    }

    fn label(self) -> String {
        match self {
            Order::RoundRobin => "round-robin".to_string(),
            Order::Burst(run_len) => format!("burst x{run_len}"),
        }
    }
}

/// Run `turns` turns across `owners` owners against one store.
fn run(
    fixture: &Fixture,
    owners: usize,
    turns: usize,
    caps: Caps,
    policy: Policy,
    queue_blocks_per_turn: u32,
    order: Order,
) -> Metrics {
    assert!(owners >= 1 && owners <= OWNERS.len());
    let block_size = fixture.block_size;
    // Owner sequence first: an order does not have to hand out turns evenly, so
    // the longest token stream has to come from the schedule rather than from
    // `turns / owners`.
    let schedule: Vec<usize> = (0..turns)
        .map(|turn| order.owner_at(turn, owners))
        .collect();
    let max_own_turns = (0..owners)
        .map(|owner| schedule.iter().filter(|&&o| o == owner).count())
        .max()
        .unwrap_or(0);
    let longest = fixture.total_len(max_own_turns.saturating_sub(1)) as usize;
    // Distinct high bits per owner, so no two owners share a block hash chain
    // and the lineage predicates keep them apart for real.
    let streams: Vec<Vec<u32>> = (0..owners)
        .map(|owner| {
            (0..longest as u32)
                .map(|i| ((owner as u32) << 24) | i)
                .collect()
        })
        .collect();
    let extra_keys = vec![Vec::new(); longest.div_ceil(block_size as usize)];
    let root = OWNERS[0];

    let mut store: VecDeque<Lineage> = VecDeque::new();
    let mut own_turns = vec![0usize; owners];
    // The model holds ONE live paged request and ONE history checkpoint slot, so
    // both belong to whichever owner ran last.
    let mut live_owner: Option<usize> = None;
    let mut history_slot: Option<(usize, u32)> = None;
    let mut persisted: HashSet<(usize, u32)> = HashSet::new();
    // Deepest boundary each owner ever got onto disk — what a restart restores.
    let mut deepest_persisted: BTreeMap<usize, u32> = BTreeMap::new();
    let mut metrics = Metrics {
        turns,
        ..Metrics::default()
    };

    for &owner in schedule.iter().take(turns) {
        let owner_id = OWNERS[owner];
        let own_turn = own_turns[owner];
        let tokens = &streams[owner];
        let prompt_len = fixture.prompt_len(own_turn);

        // 1. What the paged prefix cache hands this turn. A live continue keeps
        //    the request open and reuses its exact token count; anything else
        //    goes back through the block-aligned prefix lookup.
        let live = live_owner == Some(owner) && own_turn > 0;
        let cached_prefix_len = if own_turn == 0 {
            0
        } else if live {
            fixture.total_len(own_turn - 1)
        } else {
            align_down(fixture.total_len(own_turn - 1), block_size)
                .min(align_down(prompt_len - 1, block_size))
        };
        metrics.cached_prefix_tokens += u64::from(cached_prefix_len);

        // 2. Where the GDN recurrent state for that prefix comes from, in the
        //    order `prepare_dense_gdn_prefix_state` tries the sources. The
        //    cold-sidecar arm below the store cannot fire in-process: a hot
        //    prefix-cache hit takes the restore walk's `idx >= full_blocks`
        //    exit, so `take_restored_sidecar` is always `None` here.
        if cached_prefix_len == 0 {
            // Cold start: nothing primed, nothing replayed.
        } else if live {
            metrics.warm_live += 1;
        } else if history_slot == Some((owner, cached_prefix_len)) {
            metrics.warm_history_slot += 1;
        } else if let Some(idx) = find_longest_valid_gdn_checkpoint_index(
            &store,
            owner_id,
            tokens,
            cached_prefix_len,
            block_size,
            &extra_keys,
            0,
            |_| true,
        ) {
            let restored = store[idx].prefix_len;
            let replayed = cached_prefix_len - restored;
            metrics.gdn_replay_tokens += u64::from(replayed);
            if replayed == 0 {
                metrics.warm_store_exact += 1;
            } else {
                metrics.warm_store_partial += 1;
            }
            // A successful lookup is an LRU touch, and the touch moves the entry
            // to the back, which is where every `find`-based eviction arm looks
            // last. Skipping it would flatter the store.
            let touched = store.remove(idx).expect("index came from this store");
            store.push_back(touched);
        } else {
            metrics.gdn_replay_tokens += u64::from(cached_prefix_len);
            metrics.warm_full_replay += 1;
        }

        // 3. This turn's prefill publishes its ladder, one rung at a time, with
        //    a prune after each.
        for boundary in
            gdn_prefill_checkpoint_boundaries(prompt_len as usize, cached_prefix_len, block_size)
        {
            store.retain(|existing| {
                !(existing.owner == owner_id
                    && existing.prefix_len == boundary
                    && existing.block_size == block_size)
            });
            store.push_back(paged_lineage(
                "prefill",
                owner_id,
                tokens,
                boundary,
                block_size,
                &extra_keys,
                0,
            ));
            prune_gdn_checkpoints(
                &mut store,
                GdnRetentionCaps::ladder(caps.limit, caps.per_owner),
                root,
                policy.active_owner(owner_id),
            );
        }

        // 4. End of turn: the K/V chain has ratcheted forward and the capture
        //    asks the store where a sidecar may be anchored.
        let total_len = fixture.total_len(own_turn);
        let full_blocks = total_len / block_size;
        let chain_blocks = (queue_blocks_per_turn * (own_turn as u32 + 1)).min(full_blocks);
        if chain_blocks == 0 {
            metrics.cold_chain_empty += 1;
        } else {
            match select_gdn_sidecar_boundary(
                &store,
                owner_id,
                tokens,
                chain_blocks * block_size,
                block_size,
                &extra_keys,
                0,
                |_| true,
            ) {
                GdnSidecarBoundary::Selected { boundary, .. } => {
                    if persisted.insert((owner, boundary)) {
                        metrics.cold_captures += 1;
                    } else {
                        metrics.cold_dedup_skips += 1;
                    }
                    let depth = deepest_persisted.entry(owner).or_default();
                    *depth = (*depth).max(boundary);
                }
                GdnSidecarBoundary::Missed(_) => metrics.cold_boundary_misses += 1,
            }
        }

        history_slot = Some((owner, total_len));
        live_owner = Some(owner);
        own_turns[owner] += 1;
    }

    metrics.restore_depth_tokens = deepest_persisted.values().map(|d| u64::from(*d)).sum();
    metrics.restore_depth_min = (0..owners)
        .map(|owner| deepest_persisted.get(&owner).copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    metrics
}

const SWEEP_TURNS: usize = 40;

/// Concurrent owners the product actually produces: the root Pi session plus
/// the four child loops the subagent extension caps concurrency at.
///
/// A LITERAL, deliberately — NOT `GDN_PREFIX_CHECKPOINT_LIMIT`. The blind-turn
/// cliff sits exactly one owner past the cap, so a loop bounded by the cap moves
/// with any shrink of it and can never reach the cliff: swept at
/// `1..=GDN_PREFIX_CHECKPOINT_LIMIT`, the mutation `cap 5 -> 4` (with per-owner
/// 4 -> 3, which the `const _` assert below the cap forces) leaves this test
/// GREEN. Measured, not reasoned. This constant is the demand; the cap is the
/// supply, and only a fixed demand can catch the supply falling.
const SIZED_FLEET_OWNERS: usize = 5;
const SWEEP_OWNERS: [usize; 6] = [1, 2, 3, 4, 5, 6];

/// Capacity variants. `limit 24` is the upper bound for this sweep: six owners
/// times a per-owner cap of four is 24, so the global cap never fires and both
/// arms are the same code path.
const CAPACITY_SWEEP: [Caps; 9] = [
    Caps::SHIPPED,
    Caps {
        limit: 6,
        per_owner: 4,
    },
    Caps {
        limit: 7,
        per_owner: 4,
    },
    // The implicit-root call site on a PERSIST turn: `gdn_retention_caps`
    // hands the per-owner cap over as the GLOBAL limit when no root owner was
    // named, so limit and per-owner are both 4 there. The persist-OFF pair
    // (2, 2) is deliberately absent — this sim measures cold-sidecar coverage,
    // and a turn with no cold policy writes no sidecar to measure.
    Caps {
        limit: 4,
        per_owner: 4,
    },
    // HEAD's per-owner cap, before the ladder widened it.
    Caps {
        limit: 5,
        per_owner: 2,
    },
    Caps {
        limit: 9,
        per_owner: 4,
    },
    Caps {
        limit: 13,
        per_owner: 4,
    },
    Caps {
        limit: 13,
        per_owner: 2,
    },
    Caps {
        limit: 24,
        per_owner: 4,
    },
];

fn sweep(fixture: &Fixture, caps: Caps, policy: Policy, queue: u32, order: Order) -> Vec<Metrics> {
    SWEEP_OWNERS
        .iter()
        .map(|&owners| run(fixture, owners, SWEEP_TURNS, caps, policy, queue, order))
        .collect()
}

/// The table that decided the policy: every counter, both arms, for one
/// (fixture, caps, order).
fn print_decision_table(fixture: &Fixture, caps: Caps, queue: u32, order: Order) {
    println!(
        "\n{}  {} turns {}, block {}, chain +{} blocks/turn, limit {} per-owner {}",
        fixture.label,
        SWEEP_TURNS,
        order.label(),
        fixture.block_size,
        queue,
        caps.limit,
        caps.per_owner
    );
    println!(
        "  {:<26} {:>2} {:>6} {:>6} {:>5} {:>8} {:>6} {:>10} {:>9}",
        "policy", "n", "WRITE", "dedup", "miss", "replay%", "blind", "restoreTok", "worstOwn"
    );
    for policy in [Policy::ProtectsPublisher, Policy::Unrestricted] {
        for (row, owners) in sweep(fixture, caps, policy, queue, order)
            .into_iter()
            .zip(SWEEP_OWNERS)
        {
            println!(
                "  {:<26} {:>2} {:>6} {:>6} {:>5} {:>7.1}% {:>6} {:>10} {:>9}",
                policy.label(),
                owners,
                row.cold_captures,
                row.cold_dedup_skips,
                row.cold_boundary_misses,
                row.replay_fraction() * 100.0,
                row.warm_full_replay,
                row.restore_depth_tokens,
                row.restore_depth_min,
            );
        }
    }
}

/// Everything the two reviewers said the sweep never looked at.
#[test]
fn decision_tables() {
    for order in [Order::RoundRobin, Order::Burst(4)] {
        for fixture in [Fixture::AGENT, Fixture::CHAT, Fixture::PRIOR_PASS] {
            for caps in [
                Caps::SHIPPED,
                Caps {
                    limit: 4,
                    per_owner: 4,
                },
                Caps {
                    limit: 5,
                    per_owner: 2,
                },
            ] {
                print_decision_table(&fixture, caps, QUEUE_BLOCKS_PER_TURN, order);
            }
        }
    }
}

/// Every counter, one line per owner count. Used for the shipped caps, which is
/// the configuration a reader has to be able to reason about.
fn print_detail(fixture: &Fixture, caps: Caps, queue: u32) {
    println!(
        "\n{}  {} turns round-robin, block {}, chain +{} blocks/turn, limit {} per-owner {}",
        fixture.label, SWEEP_TURNS, fixture.block_size, queue, caps.limit, caps.per_owner
    );
    println!(
        "  {:<26} {:>2} {:>6} {:>6} {:>5} {:>5} {:>8} {:>12} {:>12} {:>6}",
        "policy",
        "n",
        "WRITE",
        "dedup",
        "miss",
        "none",
        "replay%",
        "replay tok",
        "cached tok",
        "blind"
    );
    for policy in [Policy::ProtectsPublisher, Policy::Unrestricted] {
        for (row, owners) in sweep(fixture, caps, policy, queue, Order::RoundRobin)
            .into_iter()
            .zip(SWEEP_OWNERS)
        {
            println!(
                "  {:<26} {:>2} {:>6} {:>6} {:>5} {:>5} {:>7.1}% {:>12} {:>12} {:>6}",
                policy.label(),
                owners,
                row.cold_captures,
                row.cold_dedup_skips,
                row.cold_boundary_misses,
                row.cold_chain_empty,
                row.replay_fraction() * 100.0,
                row.gdn_replay_tokens,
                row.cached_prefix_tokens,
                row.warm_full_replay,
            );
        }
    }
}

/// One line per arm: sidecars written, then hot-path replay share, across the
/// owner sweep. Used for the capacity variants.
fn print_compact(fixture: &Fixture, caps: Caps, queue: u32) {
    println!(
        "  limit {:>2} per-owner {}   {:>5.0} MiB of 27B checkpoints at the cap",
        caps.limit,
        caps.per_owner,
        mib(checkpoint_bytes(&dense_27b()) * caps.limit),
    );
    for policy in [Policy::ProtectsPublisher, Policy::Unrestricted] {
        let rows = sweep(fixture, caps, policy, queue, Order::RoundRobin);
        let captures: Vec<String> = rows
            .iter()
            .map(|row| format!("{:>4}", row.cold_captures))
            .collect();
        let replay: Vec<String> = rows
            .iter()
            .map(|row| format!("{:>5.1}", row.replay_fraction() * 100.0))
            .collect();
        println!(
            "    {}  WRITE {}   replay% {}",
            policy.label(),
            captures.join(" "),
            replay.join(" ")
        );
    }
}

fn owner_axis() -> String {
    SWEEP_OWNERS
        .iter()
        .map(|n| format!("{n:>4}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Bytes one resident checkpoint costs on `config`, sized by the same geometry
/// the on-disk sidecar is sized by.
fn checkpoint_bytes(config: &Qwen3_5Config) -> usize {
    let geo = gdn_sidecar::geometry(config, "BFloat16").expect("geometry");
    let layout = gdn_sidecar::layout_at(&geo, 4096);
    layout.bytes_per_tensor * layout.num_layers as usize
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

/// Qwen3.6-27B dense, the shape the 75.00 MiB figure in `paged_forward` comes
/// from.
fn dense_27b() -> Qwen3_5Config {
    Qwen3_5Config {
        num_layers: 64,
        full_attention_interval: 4,
        linear_conv_kernel_dim: 4,
        linear_num_key_heads: 16,
        linear_key_head_dim: 128,
        linear_num_value_heads: 48,
        linear_value_head_dim: 128,
        ..small_config_shell()
    }
}

/// `Qwen/Qwen3.5-0.8B`, verbatim from its `text_config`.
fn dense_0_8b() -> Qwen3_5Config {
    Qwen3_5Config {
        num_layers: 24,
        full_attention_interval: 4,
        linear_conv_kernel_dim: 4,
        linear_num_key_heads: 16,
        linear_key_head_dim: 128,
        linear_num_value_heads: 16,
        linear_value_head_dim: 128,
        ..small_config_shell()
    }
}

/// Fields `gdn_sidecar::geometry` never reads, filled with anything valid.
fn small_config_shell() -> Qwen3_5Config {
    Qwen3_5Config {
        vocab_size: 32,
        hidden_size: 16,
        num_layers: 8,
        num_heads: 2,
        num_kv_heads: 1,
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        head_dim: 8,
        tie_word_embeddings: false,
        attention_bias: false,
        max_position_embeddings: 256,
        pad_token_id: 0,
        eos_token_id: 1,
        bos_token_id: 2,
        linear_num_value_heads: 2,
        linear_num_key_heads: 1,
        linear_key_head_dim: 3,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        use_block_paged_cache: None,
        persist_paged_cache: None,
        n_mtp_layers: 0,
    }
}

/// The measurement. Prints every arm under `--nocapture`; the assertions at the
/// end are the floors that keep the numbers from rotting.
#[test]
fn multi_owner_sweep_measures_both_consumers() {
    println!("\nowners axis for every table below: {}", owner_axis());
    for fixture in [Fixture::AGENT, Fixture::CHAT, Fixture::PRIOR_PASS] {
        print_detail(&fixture, Caps::SHIPPED, QUEUE_BLOCKS_PER_TURN);
        println!("\n{}  capacity sweep", fixture.label);
        for caps in CAPACITY_SWEEP {
            print_compact(&fixture, caps, QUEUE_BLOCKS_PER_TURN);
        }
    }

    let bytes_27b = checkpoint_bytes(&dense_27b());
    let bytes_0_8b = checkpoint_bytes(&dense_0_8b());
    println!(
        "\none checkpoint: 27B dense {} B ({:.2} MiB)   0.8B dense {} B ({:.2} MiB)   ratio {:.1}x",
        bytes_27b,
        mib(bytes_27b),
        bytes_0_8b,
        mib(bytes_0_8b),
        bytes_27b as f64 / bytes_0_8b as f64
    );
    println!(
        "a {}-slot count bound therefore reserves {:.0} MiB on the 27B and {:.0} MiB on the 0.8B",
        GDN_PREFIX_CHECKPOINT_LIMIT,
        mib(bytes_27b * GDN_PREFIX_CHECKPOINT_LIMIT),
        mib(bytes_0_8b * GDN_PREFIX_CHECKPOINT_LIMIT)
    );

    let agent_1 = run(
        &Fixture::AGENT,
        1,
        SWEEP_TURNS,
        Caps::SHIPPED,
        Policy::ProtectsPublisher,
        QUEUE_BLOCKS_PER_TURN,
        Order::RoundRobin,
    );
    assert!(
        agent_1.cold_selectable() >= AGENT_SELECTABLE_FLOOR,
        "one agent owner found an anchorable rung on {} of {SWEEP_TURNS} turns, \
         floor {AGENT_SELECTABLE_FLOOR}: {agent_1:?}",
        agent_1.cold_selectable()
    );

    for owners in 1..=SIZED_FLEET_OWNERS {
        let small_fleet = run(
            &Fixture::AGENT,
            owners,
            SWEEP_TURNS,
            Caps::SHIPPED,
            Policy::ProtectsPublisher,
            QUEUE_BLOCKS_PER_TURN,
            Order::RoundRobin,
        );
        assert_eq!(
            small_fleet.warm_full_replay, SMALL_FLEET_BLIND_TURNS,
            "{owners} agent owners left {} of {SWEEP_TURNS} turns with no checkpoint at all, \
             expected {SMALL_FLEET_BLIND_TURNS}: {small_fleet:?}",
            small_fleet.warm_full_replay
        );
        assert!(
            small_fleet.replay_fraction() <= SMALL_FLEET_REPLAY_CEILING,
            "{owners} agent owners re-forwarded {:.1}% of every cached prefix, \
             ceiling {:.1}%: {small_fleet:?}",
            small_fleet.replay_fraction() * 100.0,
            SMALL_FLEET_REPLAY_CEILING * 100.0
        );
    }

    for owners in 1..=2 {
        let chat = run(
            &Fixture::CHAT,
            owners,
            SWEEP_TURNS,
            Caps::SHIPPED,
            Policy::ProtectsPublisher,
            QUEUE_BLOCKS_PER_TURN,
            Order::RoundRobin,
        );
        assert!(
            chat.cold_captures >= CHAT_CAPTURE_FLOOR,
            "{owners} chat owners wrote {} sidecars over {SWEEP_TURNS} turns, \
             floor {CHAT_CAPTURE_FLOOR}: {chat:?}",
            chat.cold_captures
        );
    }
}

/// Turns on which one agent owner had a retained rung sitting under the
/// persisted chain's reach. Measured 40 of 40. Counting selectability rather
/// than writes keeps this floor off the `contains_in` dedup, which on this
/// fixture swallows all but three of the forty.
///
/// Catches, verified by mutation: `GDN_PREFIX_CHECKPOINTS_PER_OWNER` 4 -> 1
/// takes this to 0 of 40. Keeping one rung per owner keeps the DEEPEST, the
/// chain advances ~130 tokens a turn, and the prompt starts at 1024, so nothing
/// ever sits under the reach.
///
/// Does NOT catch `GDN_CHECKPOINT_LADDER_RUNGS` 4 -> 1, which only takes it to
/// 33: the per-owner victim rule keeps the widest-spanned rungs, so even a
/// single-rung ladder leaves the first turn's shallow boundary resident for most
/// of the conversation. `selectable_is_not_written` is the test that fires on
/// that mutation.
const AGENT_SELECTABLE_FLOOR: usize = 30;

/// Agent turns that found no checkpoint at all and re-forwarded the whole cached
/// prefix. Measured 0 at one, two and three owners, so this is an equality: up
/// to three owners nobody should ever go blind, and one blind turn is not a
/// rounding error but an owner losing its last rung permanently (the F1 path
/// does not reset `cached_prefix_len`, so the miss repeats). Still 0 at four and
/// five owners; at SIX it is 28 of 40, under either arm — that cliff is the
/// count bound, not the victim search.
///
/// The loop that reads this sweeps `1..=SIZED_FLEET_OWNERS` — a fixed 5, the
/// fleet the product produces — so this assertion is what pins the CAP at the
/// count that justifies its value. Bounding the loop by
/// `GDN_PREFIX_CHECKPOINT_LIMIT` instead does not work and was measured not to:
/// the cliff sits one owner past the cap, so the loop moves with the mutation
/// and never reaches it.
///
/// Catches, verified by mutation: replacing the global-cap victim search with
/// plain FIFO (`checkpoints.remove(0)`) blinds a turn at three owners.
/// `GDN_PREFIX_CHECKPOINT_LIMIT` 5 -> 4 (with `GDN_PREFIX_CHECKPOINTS_PER_OWNER`
/// 4 -> 3, which the `const _` assert beside the cap forces) blinds 28 of 40 at
/// five owners, `cold_captures` 23 and `warm_store_partial` 7.
const SMALL_FLEET_BLIND_TURNS: usize = 0;

/// Share of every cached prefix token those same runs re-forward. Measured 0.0%,
/// 5.9% and 7.8%. The residual is the ladder's granularity: a turn anchors at
/// the deepest rung at or below its cached prefix and replays the remainder.
///
/// Catches, verified by mutation: flipping
/// `find_longest_valid_gdn_checkpoint_index` to prefer the SHALLOWEST valid
/// checkpoint instead of the deepest takes two agent owners to 98.6%. That
/// mutation leaves `cold_captures` and the blind-turn count untouched, so this
/// is the only assertion here that sees it — which is the argument for metric B
/// existing at all.
const SMALL_FLEET_REPLAY_CEILING: f64 = 0.15;

/// Sidecars one or two chat owners write. Measured 39 and 40 — this fixture's
/// chain overtakes its prompt, so nearly every turn anchors somewhere new. The
/// floor is half of that, which is the "nobody would notice if the hit rate
/// halved" case stated plainly.
///
/// Catches, verified by mutation: the same shallowest-instead-of-deepest flip
/// takes these to 1 and 2. Every selection then lands on the same shallow rung
/// and `contains_in` swallows all but the first.
///
/// This one is deliberately insensitive to eviction: it survives
/// `GDN_PREFIX_CHECKPOINTS_PER_OWNER` 4 -> 1 (36, 32),
/// `GDN_CHECKPOINT_LADDER_RATIO` 4 -> 64 (39, 38) and a FIFO global victim
/// (39, 40), because a chain that has overtaken its prompt anchors on the
/// deepest rung and needs no ladder. That is the point of pairing it with the
/// agent fixture rather than replacing it.
const CHAT_CAPTURE_FLOOR: usize = 20;

/// The standalone simulator this measurement replaces reported a different
/// quantity under the name "hits": it counted every turn where a rung sat under
/// the chain's reach, with no `contains_in` check, so a store frozen on one
/// boundary scored a hit on all forty turns. Production writes a sidecar only
/// when that exact chain is not already on disk.
///
/// Printing both columns side by side shows which of the two numbers each table
/// row was, so nobody has to guess later.
#[test]
fn selectable_is_not_written() {
    println!("\nowners axis: {}", owner_axis());
    for fixture in [Fixture::PRIOR_PASS, Fixture::AGENT] {
        println!("\n{}  shipped caps", fixture.label);
        for policy in [Policy::ProtectsPublisher, Policy::Unrestricted] {
            let rows = sweep(
                &fixture,
                Caps::SHIPPED,
                policy,
                QUEUE_BLOCKS_PER_TURN,
                Order::RoundRobin,
            );
            let selectable: Vec<String> = rows
                .iter()
                .map(|row| format!("{:>4}", row.cold_selectable()))
                .collect();
            let written: Vec<String> = rows
                .iter()
                .map(|row| format!("{:>4}", row.cold_captures))
                .collect();
            println!(
                "  {}  selectable {}   WRITE {}",
                policy.label(),
                selectable.join(" "),
                written.join(" ")
            );
        }
    }

    // A single owner on the prior fixture is the clearest case: the chain never
    // outruns a prompt that grows by 2048 tokens a turn, so the same shallow
    // rung is selected turn after turn and only the first selection writes.
    //
    // The equality is also the ladder's own regression floor. Mutating
    // `GDN_CHECKPOINT_LADDER_RUNGS` 4 -> 1 takes it from 40 to 9 and takes the
    // sidecars written from 3 to 1 — on this fixture a single-rung ladder is
    // most of the cold tier gone.
    let single = run(
        &Fixture::PRIOR_PASS,
        1,
        SWEEP_TURNS,
        Caps::SHIPPED,
        Policy::ProtectsPublisher,
        QUEUE_BLOCKS_PER_TURN,
        Order::RoundRobin,
    );
    assert_eq!(single.cold_selectable(), SWEEP_TURNS);
    assert!(
        single.cold_captures * 4 < single.cold_selectable(),
        "selectable {} vs written {}: the dedup is supposed to swallow most \
         selections on this fixture, so a capture counter without it overstates \
         metric A by a large factor",
        single.cold_selectable(),
        single.cold_captures,
    );
}

/// The control arm is the shipped code with one argument changed, so it has to
/// be the pre-fix policy exactly, not approximately.
///
/// `prune_gdn_checkpoints` reads `active_owner_id` in exactly one place — the
/// `OwnerScope::Excluding` it hands the first [`redundant_victim`]. `Excluding(x)`
/// and `Any` differ only in `admits`, and only for entries owned by `x`. An `x`
/// the store does not hold therefore makes the two scopes indistinguishable, so
/// arm 1 returns exactly what arm 2 would and arm 2 never decides anything: the
/// three-arm loop collapses onto the unrestricted two-arm search HEAD had.
#[test]
fn an_absent_active_owner_is_the_unrestricted_arm() {
    let (tokens, keys) = ladder_tokens_and_keys();
    let mut store = VecDeque::new();
    for (owner, boundaries) in [
        ("owner-0", &FULL_LADDER[..]),
        ("owner-1", &[48, 4080][..]),
        ("owner-2", &[4080][..]),
    ] {
        for &boundary in boundaries {
            store.push_back(paged_lineage(
                "resident",
                owner,
                &tokens,
                boundary,
                LADDER_BLOCK_SIZE,
                &keys,
                0,
            ));
        }
    }

    assert!(
        store
            .iter()
            .all(|checkpoint| checkpoint.owner != ABSENT_ACTIVE_OWNER),
        "the sentinel must own nothing"
    );
    for checkpoint in &store {
        assert_eq!(
            OwnerScope::Excluding(ABSENT_ACTIVE_OWNER).admits(checkpoint.owner),
            OwnerScope::Any.admits(checkpoint.owner),
        );
    }
    assert_eq!(
        redundant_victim(&store, OwnerScope::Excluding(ABSENT_ACTIVE_OWNER)),
        redundant_victim(&store, OwnerScope::Any),
    );
}

/// The two arms must actually disagree, or the sweep is measuring one policy
/// twice. This is `one_subagent_turn_leaves_every_sibling_a_checkpoint`'s store,
/// run through both, and it shows exactly what the disagreement is: who pays the
/// coverage. The publisher keeps a shallow rung under the shipped arm and the
/// root keeps one under the control arm — and under BOTH, every owner keeps a
/// checkpoint. The choice moves metric A around and cannot move metric B, which
/// is the whole finding this file was built to check.
#[test]
fn the_two_arms_disagree_about_who_pays() {
    let resident: [(&'static str, &'static [u32]); 3] = [
        ("owner-0", &[48, 4080]),
        ("owner-1", &[4080]),
        ("owner-2", &[4080]),
    ];

    let mut applied = VecDeque::new();
    seed_foreign_owners(
        &mut applied,
        &resident,
        "owner-0",
        GDN_PREFIX_CHECKPOINT_LIMIT,
    );
    publish_ladder(
        &mut applied,
        "owner-3",
        "owner-0",
        GDN_PREFIX_CHECKPOINT_LIMIT,
    );

    let mut control = VecDeque::new();
    seed_foreign_owners(
        &mut control,
        &resident,
        "owner-0",
        GDN_PREFIX_CHECKPOINT_LIMIT,
    );
    let (tokens, keys) = ladder_tokens_and_keys();
    for boundary in FULL_LADDER {
        control.push_back(paged_lineage(
            "prefill",
            "owner-3",
            &tokens,
            boundary,
            LADDER_BLOCK_SIZE,
            &keys,
            0,
        ));
        prune_gdn_checkpoints(
            &mut control,
            GdnRetentionCaps::ladder(
                GDN_PREFIX_CHECKPOINT_LIMIT,
                GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            ),
            "owner-0",
            ABSENT_ACTIVE_OWNER,
        );
    }

    // Shipped: the root's spare rung pays, so the publisher keeps a shallow one
    // and can anchor a capture at the end of this same turn.
    assert_eq!(
        retained_boundaries(&applied, "owner-3"),
        vec![48, 4080],
        "{:?}",
        store_shape(&applied)
    );
    assert_eq!(
        retained_boundaries(&applied, "owner-0"),
        vec![4080],
        "{:?}",
        store_shape(&applied)
    );

    // Control: the publisher's own ladder pays instead and collapses to the
    // single endpoint rung, while the root keeps the spare.
    assert_eq!(
        retained_boundaries(&control, "owner-3"),
        vec![4080],
        "{:?}",
        store_shape(&control)
    );
    assert_eq!(
        retained_boundaries(&control, "owner-0"),
        vec![48, 4080],
        "{:?}",
        store_shape(&control)
    );

    // And the part that does NOT differ: neither arm blinds anyone here.
    for store in [&applied, &control] {
        for owner in ["owner-0", "owner-1", "owner-2", "owner-3"] {
            assert!(
                !retained_boundaries(store, owner).is_empty(),
                "{owner}: {:?}",
                store_shape(store)
            );
        }
    }
}

/// The decision, as an assertion: preferring a foreign victim costs the hot path
/// NOTHING. Not "little" — nothing, in every cell of the sweep.
///
/// The reason is structural, so a counter-example would mean the argument in
/// `prune_gdn_checkpoints`'s doc is wrong. [`redundant_victim`] never returns an
/// entry whose owner would be left empty, so arms 1 and 2 draw from one
/// candidate set and only reorder it. The blinding arm is reached exactly when
/// that set is empty — every owner down to one entry, still over cap — and the
/// number of free evictions available before that, `len - distinct_owners`, does
/// not depend on the order they are taken in.
///
/// Metric B is the one a user feels: a re-forwarded prefix is GPU time on the
/// turn they are waiting for, while metric A is an SSD write only a restart
/// reads. Equality here is what makes the sidecar gain free.
#[test]
fn the_publisher_arm_never_costs_hot_path_replay() {
    let mut cells = 0usize;
    for order in [Order::RoundRobin, Order::Burst(4)] {
        for fixture in [Fixture::AGENT, Fixture::CHAT, Fixture::PRIOR_PASS] {
            for caps in CAPACITY_SWEEP {
                for owners in SWEEP_OWNERS {
                    let applied = run(
                        &fixture,
                        owners,
                        SWEEP_TURNS,
                        caps,
                        Policy::ProtectsPublisher,
                        QUEUE_BLOCKS_PER_TURN,
                        order,
                    );
                    let control = run(
                        &fixture,
                        owners,
                        SWEEP_TURNS,
                        caps,
                        Policy::Unrestricted,
                        QUEUE_BLOCKS_PER_TURN,
                        order,
                    );
                    assert_eq!(
                        (applied.gdn_replay_tokens, applied.warm_full_replay),
                        (control.gdn_replay_tokens, control.warm_full_replay),
                        "{} {} n={owners} limit {} per-owner {}: the arms disagree \
                         about the HOT path, which the structural argument in \
                         `prune_gdn_checkpoints` says cannot happen\n  applied \
                         {applied:?}\n  control {control:?}",
                        fixture.label,
                        order.label(),
                        caps.limit,
                        caps.per_owner,
                    );
                    cells += 1;
                }
            }
        }
    }
    println!("hot path identical in all {cells} cells");
    assert_eq!(
        cells,
        2 * 3 * CAPACITY_SWEEP.len() * SWEEP_OWNERS.len(),
        "the sweep shrank"
    );
}

/// What the free reordering buys: sidecars. Summed over the owner sweep, because
/// per-owner-count the two arms cross (at two agent owners the control writes
/// one more, then falls behind and stays behind).
///
/// Burst order is the shape pi actually produces — four task loops interleaved
/// by the host, not a tidy rotation — and it is where the gap is widest.
///
/// Only the AGENT fixture can still show a gap at the shipped chain speed. CHAT
/// adds 96 tokens (6 blocks) per turn against a chain that ratchets
/// [`QUEUE_BLOCKS_PER_TURN`], so the persisted chain overtakes the prompt within
/// two turns and BOTH arms write a sidecar on every single turn — the ceiling,
/// `SWEEP_TURNS * SWEEP_OWNERS.len()`. A tie there is saturation, not the policy
/// failing to buy anything, so the assertion splits: at the ceiling the applied
/// arm must match it, below the ceiling it must strictly beat the control. That
/// split is what makes the CHAT arm still able to fail — it goes red if the
/// reordering ever drops a sidecar the unrestricted search kept.
#[test]
fn the_publisher_arm_buys_cold_sidecars() {
    let ceiling = SWEEP_TURNS * SWEEP_OWNERS.len();
    for order in [Order::RoundRobin, Order::Burst(4)] {
        for fixture in [Fixture::AGENT, Fixture::CHAT] {
            let total = |policy| -> usize {
                sweep(
                    &fixture,
                    Caps::SHIPPED,
                    policy,
                    QUEUE_BLOCKS_PER_TURN,
                    order,
                )
                .iter()
                .map(|row| row.cold_captures)
                .sum()
            };
            let applied = total(Policy::ProtectsPublisher);
            let control = total(Policy::Unrestricted);
            println!(
                "{} {}: {applied} sidecars vs {control} (ceiling {ceiling})",
                fixture.label,
                order.label()
            );
            if control == ceiling {
                assert_eq!(
                    applied,
                    ceiling,
                    "{} {}: the unrestricted search already wrote a sidecar on \
                     every turn, so preferring a foreign victim must not lose \
                     one — it wrote {applied}",
                    fixture.label,
                    order.label(),
                );
            } else {
                assert!(
                    applied > control,
                    "{} {}: preferring a foreign victim wrote {applied} sidecars \
                     over the owner sweep and the unrestricted search wrote \
                     {control}, so it is buying nothing and should be reverted",
                    fixture.label,
                    order.label(),
                );
            }
        }
    }
}

/// One owner is one policy. `Excluding(x)` admits every entry the store holds
/// when `x` owns none of them, and a single-owner store is the ONLY entry `x`
/// owns — so the first arm returns `None` on every iteration and the second arm
/// is the whole search. ChatSession, the server's single-session path and a
/// non-delegating agent all live here, and none of them may pay for this.
#[test]
fn one_owner_is_the_same_policy_either_way() {
    for order in [Order::RoundRobin, Order::Burst(4)] {
        for fixture in [Fixture::AGENT, Fixture::CHAT, Fixture::PRIOR_PASS] {
            for caps in CAPACITY_SWEEP {
                let applied = run(
                    &fixture,
                    1,
                    SWEEP_TURNS,
                    caps,
                    Policy::ProtectsPublisher,
                    QUEUE_BLOCKS_PER_TURN,
                    order,
                );
                let control = run(
                    &fixture,
                    1,
                    SWEEP_TURNS,
                    caps,
                    Policy::Unrestricted,
                    QUEUE_BLOCKS_PER_TURN,
                    order,
                );
                assert_eq!(
                    applied,
                    control,
                    "{} {} limit {} per-owner {}: the single-owner path changed",
                    fixture.label,
                    order.label(),
                    caps.limit,
                    caps.per_owner,
                );
            }
        }
    }
}

/// The writer's ratchet is the one modelled number here, so check the sweep's
/// ranking does not hang off its exact value.
///
/// The ranking must never INVERT at any swept speed, and it must still be a
/// ranking at SOME of them. Those are two assertions on purpose. At the shipped
/// budget the chain covers this fixture's whole prompt on the turn it opens, so
/// both arms capture every checkpoint and the arms are genuinely tied — a real
/// result, and the one this change was made to produce, but a tie proves
/// nothing about retention. Demanding a strict win at every speed would fail on
/// that tie; accepting ties everywhere would let a policy that does nothing
/// pass. So: `>=` at every speed, and strictly greater at at least one.
#[test]
fn chain_speed_changes_the_level_not_the_ranking() {
    let mut strict_wins = 0usize;
    for queue in [
        PRE_FSYNC_BLOCKS_PER_TURN,
        QUEUE_BLOCKS_PER_TURN / 8,
        QUEUE_BLOCKS_PER_TURN / 2,
        QUEUE_BLOCKS_PER_TURN * 2,
    ] {
        let mut line = Vec::new();
        let mut applied_total = 0usize;
        let mut control_total = 0usize;
        for owners in SWEEP_OWNERS {
            let applied = run(
                &Fixture::AGENT,
                owners,
                SWEEP_TURNS,
                Caps::SHIPPED,
                Policy::ProtectsPublisher,
                queue,
                Order::RoundRobin,
            );
            let control = run(
                &Fixture::AGENT,
                owners,
                SWEEP_TURNS,
                Caps::SHIPPED,
                Policy::Unrestricted,
                queue,
                Order::RoundRobin,
            );
            assert_eq!(
                (applied.gdn_replay_tokens, applied.warm_full_replay),
                (control.gdn_replay_tokens, control.warm_full_replay),
                "queue {queue} n={owners}: the arms disagree about the hot path\n  \
                 applied {applied:?}\n  control {control:?}",
            );
            applied_total += applied.cold_captures;
            control_total += control.cold_captures;
            line.push(format!(
                "n={owners} {}/{} captures  {:.1}/{:.1} replay%",
                applied.cold_captures,
                control.cold_captures,
                applied.replay_fraction() * 100.0,
                control.replay_fraction() * 100.0
            ));
        }
        println!(
            "queue {queue} blocks/turn: {}{}",
            line.join("   "),
            if applied_total == control_total {
                "   [tied: the chain covers the fixture from turn 1]"
            } else {
                ""
            }
        );
        assert!(
            applied_total >= control_total,
            "queue {queue}: the ranking INVERTED at this chain speed, \
             {applied_total} vs {control_total} sidecars",
        );
        if applied_total > control_total {
            strict_wins += 1;
        }
    }
    assert!(
        strict_wins > 0,
        "every swept chain speed tied, so this sweep no longer measures the retention \
         policy at all — a do-nothing policy would pass it. Add a slower speed, or \
         lengthen the fixture so the chain starts behind again."
    );
}

/// A count bound is a bound on the wrong quantity when the thing counted scales
/// with the model.
#[test]
fn a_checkpoint_slot_costs_what_the_model_it_belongs_to_costs() {
    let bytes_27b = checkpoint_bytes(&dense_27b());
    let bytes_0_8b = checkpoint_bytes(&dense_0_8b());
    assert_eq!(bytes_27b, 78_446_592);
    assert_eq!(bytes_0_8b, 10_100_736);
    assert!(
        bytes_27b > bytes_0_8b * 7,
        "{bytes_27b} vs {bytes_0_8b}: the two models' slots cost within 7x of each other, \
         so a count bound is no longer obviously the wrong shape"
    );
}
