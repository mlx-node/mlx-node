//! T-D0.3 gating tests: the committed-frontier settle, the atomic
//! cross-group reservation, the all-rows projection, and the paged-loop
//! tap — every one behavior-neutral for the autoregressive lane.

use super::flat_verify_tests::assert_bitwise_eq;
use super::*;
use crate::engine::spec_paged::SpecPagedCache;

/// Tiny hybrid config whose PAGED coordinator builds for real in
/// `Gemma4Inner::new`: production pool constraints require block size
/// 8/16/32 and head size >= 32, and `paged_cache_memory_mb` keeps the
/// full group's Metal pool at ~8 MiB.
fn tiny_paged_config_value() -> serde_json::Value {
    serde_json::json!({
        "vocab_size": 64,
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "intermediate_size": 16,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "max_position_embeddings": 256,
        "sliding_window": 16,
        "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ],
        "use_block_paged_cache": true,
        "paged_block_size": 8,
        "paged_cache_memory_mb": 8,
        "final_logit_softcapping": 30.0,
        "eos_token_ids": []
    })
}

fn tiny_paged_config() -> Gemma4Config {
    serde_json::from_value(tiny_paged_config_value())
        .expect("tiny paged Gemma4 config must deserialize")
}

/// [`tiny_paged_config`] with the FULL-attention group first in
/// `layer_types`, so grouping assigns it the lower group id — the
/// atomicity test's mutation (partial reservation leaking the earlier
/// group's blocks) needs the full group reserved before the sliding
/// group's exhaustion is reached.
fn tiny_paged_config_full_first() -> Gemma4Config {
    let mut value = tiny_paged_config_value();
    value["layer_types"] = serde_json::json!([
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention"
    ]);
    serde_json::from_value(value).expect("tiny paged Gemma4 config must deserialize")
}

/// Placeholder pool sized to the allocator, or `None` (skip) without a
/// Metal device — the `paged_kv_cache_adapter` test-shim pattern.
fn maybe_test_pool(num_blocks: u32, block_size: u32) -> Option<Arc<mlx_paged_attn::LayerKVPool>> {
    let cfg = mlx_paged_attn::PagedAttentionConfig {
        block_size,
        num_kv_heads: 1,
        head_size: 32,
        num_layers: 2,
        ..mlx_paged_attn::PagedAttentionConfig::default()
    };
    match mlx_paged_attn::LayerKVPool::new_for_test(
        cfg,
        num_blocks,
        2,
        mlx_paged_attn::metal::MetalDtype::Float16,
    ) {
        Ok(pool) => Some(Arc::new(pool)),
        Err(error) if error.contains("No Metal device found") => None,
        Err(error) => panic!("unexpected new_for_test failure: {error}"),
    }
}

/// Real grouped coordinator over the tiny full-first config, with
/// per-kind allocator sizes so a test can seed exhaustion in exactly one
/// group. Returns `None` (skip) without Metal.
fn maybe_grouped_coordinator(
    full_blocks: u32,
    sliding_blocks: u32,
) -> Option<Gemma4KVCacheCoordinator> {
    let config = tiny_paged_config_full_first();
    let block_size = 8u32;
    let specs = compute_layer_kv_cache_specs(&config, block_size, KVCacheDType::BFloat16)
        .expect("tiny specs must build");
    let groups = compute_layer_kv_cache_groups(
        &config,
        block_size,
        KVCacheDType::BFloat16,
        gemma4_paged_prefill_group_max_chunk(),
    )
    .expect("tiny groups must build");
    let mut adapters = Vec::with_capacity(groups.len());
    for group in &groups {
        let blocks = match group.attention_kind {
            AttentionKind::Full => full_blocks,
            AttentionKind::SlidingWindow { .. } => sliding_blocks,
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            blocks, blocks, block_size,
        )));
        let pool = maybe_test_pool(blocks, block_size)?;
        let adapter = match group.attention_kind {
            AttentionKind::Full => PagedKVCacheAdapter::new(allocator, pool, block_size),
            AttentionKind::SlidingWindow { sliding_window } => {
                PagedKVCacheAdapter::new_sliding(allocator, pool, block_size, sliding_window, 256)
            }
        }
        .expect("tiny adapter must construct");
        adapters.push(adapter);
    }
    Some(
        Gemma4KVCacheCoordinator::new(&specs, groups, adapters, 4)
            .expect("tiny coordinator must construct"),
    )
}

fn group_count(coordinator: &Gemma4KVCacheCoordinator) -> usize {
    coordinator.inner.groups().len()
}

fn allocated_per_group(coordinator: &Gemma4KVCacheCoordinator) -> Vec<usize> {
    (0..group_count(coordinator))
        .map(|group_id| {
            coordinator
                .adapter(group_id)
                .expect("group adapter")
                .num_allocated_blocks()
        })
        .collect()
}

/// Seed exhaustion in the sliding group AFTER the full group could
/// reserve: the whole call must fail as capacity exhaustion with zero net
/// block growth in EVERY group. Mutation this catches: dropping the
/// cross-group admission pre-flight, which reserves the full group's
/// blocks and then leaks them when the sliding group exhausts.
#[test]
fn reserve_rows_all_is_atomic_across_groups() {
    let Some(mut coordinator) = maybe_grouped_coordinator(64, 4) else {
        return;
    };
    // Precondition of the mutation this test exists to catch: the full
    // group must be reserved BEFORE the sliding group exhausts, i.e. it
    // must iterate first. If grouping order ever flips, fail loudly here
    // rather than silently weakening the gate.
    assert_eq!(
        coordinator
            .adapter(0)
            .expect("group 0 adapter")
            .sliding_window(),
        0,
        "test precondition: the full-attention group must have the lower group id"
    );

    coordinator.reset_scheduled_request(7).expect("reset");
    coordinator
        .record_tokens_all(7, &(0..12).collect::<Vec<_>>())
        .expect("seed 12 tokens");
    // Sliding group: 4 blocks = null + 2 recorded, 1 free. 24 lookahead
    // rows need ceil(36/8) - 2 = 3 new blocks per group.
    let before = allocated_per_group(&coordinator);
    let error = coordinator
        .reserve_rows_all(7, 24)
        .expect_err("sliding exhaustion must fail the whole reservation");
    assert!(
        error.starts_with("context_length_exceeded:"),
        "exhaustion must keep the adapter's capacity error shape, got: {error}"
    );
    assert_eq!(
        allocated_per_group(&coordinator),
        before,
        "a failed cross-group reservation must leak zero blocks in any group"
    );
    assert_eq!(
        coordinator.request_token_count_all(7).expect("token count"),
        12,
        "a failed reservation must not advance any cursor"
    );

    // The facade maps the same exhaustion onto the skip-cycle signal.
    assert!(
        !PruneOnlySpecPagedCache::new(&mut coordinator)
            .reserve_lookahead(7, 24)
            .expect("exhaustion is not an error at the facade"),
        "facade must report pool exhaustion as Ok(false)"
    );
    assert_eq!(allocated_per_group(&coordinator), before);

    // A coverable reservation lands in every group without advancing the
    // cursor, and the following verify-shaped write allocates nothing.
    let groups = group_count(&coordinator) as u32;
    let new_blocks = coordinator
        .reserve_rows_all(7, 8)
        .expect("8 lookahead rows fit every group");
    assert_eq!(new_blocks, groups, "ceil(20/8) - 2 = 1 new block per group");
    assert_eq!(
        coordinator.request_token_count_all(7).expect("token count"),
        12,
        "reservation must not advance the cursor"
    );
    let reserved = allocated_per_group(&coordinator);
    coordinator
        .record_tokens_all(7, &(12..20).collect::<Vec<_>>())
        .expect("verify write into the reserved region");
    assert_eq!(
        allocated_per_group(&coordinator),
        reserved,
        "the reserved lookahead region must absorb the verify write with zero allocation"
    );

    // Facade conformance on the same real coordinator: frontier is the
    // one count every group agrees on, commit is exact cursor arithmetic,
    // and the committed settle reaches the committed-cutoff prune (whose
    // adapter guard rejects a frontier past the cursor).
    let mut cache = PruneOnlySpecPagedCache::new(&mut coordinator);
    assert_eq!(
        cache.frontier(7),
        Some(SpecFrontier {
            attn_tokens: 20,
            recurrent_tokens: None
        })
    );
    assert_eq!(cache.frontier(99), None);
    let ticket = cache
        .record_verify(7, &[40, 41, 42])
        .expect("record verify");
    cache
        .commit_cycle(7, ticket, 1)
        .expect("commit keep=1 of 3");
    assert_eq!(
        cache.frontier(7),
        Some(SpecFrontier {
            attn_tokens: 21,
            recurrent_tokens: None
        })
    );
    cache.settle_committed(7, 21).expect("committed settle");
    assert!(
        cache
            .settle_committed(7, 22)
            .expect_err("a committed frontier past the cursor must be rejected")
            .contains("exceeds recorded token count"),
        "the committed settle must route through the committed-cutoff prune's guard"
    );
}

fn sliding_group_id(coordinator: &Gemma4KVCacheCoordinator) -> usize {
    (0..group_count(coordinator))
        .find(|&group_id| {
            coordinator
                .adapter(group_id)
                .expect("group adapter")
                .sliding_window()
                > 0
        })
        .expect("the tiny hybrid config must build a sliding group")
}

/// Cross-module gate (engine ↔ gemma4, T-D0.2 ↔ T-D0.3): the REAL
/// grouped coordinator driven through the facade call order the
/// `spec_paged` mock tests verified — reserve → record D+1 →
/// `commit_cycle(ticket, keep)` → settle at the committed frontier —
/// wrapped in `NoSettleInCycle` so an ordering law stays executable on
/// the real type. Anti-vacuity: every step is asserted through real
/// block accounting (the reservation grows each group, the verify write
/// allocates nothing, no settle work fires inside the cycle), never
/// method presence.
///
/// The checker here is the STRICTER of the two on purpose: it refuses a
/// settle of ANY basis inside an open cycle, which is the shape of a
/// driver that settles only post-commit. The identical in-cycle
/// committed-basis call is LAWFUL under `NoDurableSettleInCycle` — see
/// the sibling gates `gemma4_coordinator_is_lawful_under_the_permissive_checker`
/// and `settle_committed_never_nulls_the_rollback_range_at_coordinator_level`,
/// which drive it and assert it prunes correctly. The two verdicts are a
/// deliberate split between the two checkers, not a contradiction about
/// the coordinator.
///
/// Geometry (window 16, block 8): committed prompt 36, verify anchor +
/// 4 drafts → cursor 41, so the verify rows straddle a window edge.
///
/// Mutation this catches: a coordinator impl satisfying the trait but
/// violating the ordering laws — `record_verify` settling at the write
/// cursor mid-cycle frees sliding blocks the commit's rollback returns
/// the window into (fails the mid-cycle accounting AND the post-commit
/// live-window backing); a commit rolling back anything but exactly the
/// cycle's unaccepted rows fails the cross-group frontier equality.
#[test]
fn gemma4_coordinator_conforms_to_the_facade_call_order() {
    use crate::engine::spec_paged::NoSettleInCycle;

    let Some(mut coordinator) = maybe_grouped_coordinator(64, 16) else {
        return;
    };
    let sliding_group = sliding_group_id(&coordinator);
    coordinator.reset_scheduled_request(7).expect("reset");
    coordinator
        .record_tokens_all(7, &(0..36).collect::<Vec<_>>())
        .expect("seed the committed prompt");
    let mut cache = NoSettleInCycle::new(PruneOnlySpecPagedCache::new(&mut coordinator));

    let seeded = allocated_per_group(cache.inner().coordinator());
    assert!(
        cache.reserve_lookahead(7, 5).expect("reserve"),
        "5 lookahead rows fit every group"
    );
    let reserved = allocated_per_group(cache.inner().coordinator());
    for (group_id, (before, after)) in seeded.iter().zip(&reserved).enumerate() {
        assert_eq!(
            after - before,
            1,
            "group {group_id}: ceil(41/8) - ceil(36/8) = 1 new block"
        );
    }
    assert_eq!(
        cache
            .inner()
            .coordinator()
            .request_token_count_all(7)
            .expect("count"),
        36,
        "the reservation must not advance any group's cursor"
    );

    let ticket = cache
        .record_verify(7, &[100, 101, 102, 103, 104])
        .expect("record anchor + 4 drafts");
    assert_eq!(
        allocated_per_group(cache.inner().coordinator()),
        reserved,
        "the verify write must allocate ZERO new blocks and free none — \
         no settle work may fire inside the cycle (L-SETTLE)"
    );
    assert_eq!(
        cache
            .inner()
            .coordinator()
            .request_token_count_all(7)
            .expect("count"),
        41
    );

    // The ordering law is executable on the real coordinator too.
    let err = cache
        .settle_committed(7, 36)
        .expect_err("an in-cycle settle must trip the order check");
    assert!(err.contains("L-SETTLE"), "unexpected error text: {err}");
    assert_eq!(
        allocated_per_group(cache.inner().coordinator()),
        reserved,
        "the refused settle must never reach the coordinator"
    );

    cache
        .commit_cycle(7, ticket, 1)
        .expect("keep the anchor, roll back the 4 rejected drafts");
    assert_eq!(
        cache.frontier(7),
        Some(SpecFrontier {
            attn_tokens: 37,
            recurrent_tokens: None
        }),
        "the facade frontier must land on the committed row count"
    );
    for group_id in 0..group_count(cache.inner().coordinator()) {
        assert_eq!(
            cache
                .inner()
                .coordinator()
                .adapter(group_id)
                .expect("group adapter")
                .current_token_count_for(7),
            Some(37),
            "group {group_id} must agree on the committed frontier after the commit"
        );
    }
    assert_eq!(
        allocated_per_group(cache.inner().coordinator()),
        reserved,
        "rollback is bookkeeping-only — no block may move"
    );

    // The lawful settle, post-commit at the frontier the commit landed
    // on: cutoff 37 - 16 = 21 retires exactly sliding blocks 0-1 and
    // leaves the window the rollback returned into backed.
    cache.settle_committed(7, 37).expect("post-commit settle");
    let settled = allocated_per_group(cache.inner().coordinator());
    for (group_id, (before, after)) in reserved.iter().zip(&settled).enumerate() {
        let expected = if group_id == sliding_group { 2 } else { 0 };
        assert_eq!(
            before - after,
            expected,
            "group {group_id}: the committed settle must retire exactly the \
             out-of-window blocks"
        );
    }
    let sliding = cache
        .inner()
        .coordinator()
        .adapter(sliding_group)
        .expect("sliding adapter");
    let ids: Vec<u32> = sliding
        .block_table_for(7)
        .expect("sliding block table")
        .blocks()
        .iter()
        .map(|block| block.block_id)
        .collect();
    assert_eq!(ids[0], ids[1], "retired blocks share the null sentinel");
    assert!(
        ids[2..].iter().all(|&id| id != ids[0]),
        "the live window the commit returned into must remain physically backed"
    );
}

/// Cross-module gate (engine ↔ gemma4): the REAL coordinator under the
/// PERMISSIVE checker — the executable form of L-SETTLE as written, and
/// the only wrapper a per-chunk-settling driver can be built inside.
/// Inside an open cycle it must ADMIT the committed-basis settle (this
/// family's per-chunk settle, which is what the committed basis exists
/// for) and REFUSE the write-cursor basis, and the admitted settle must
/// reach the real prune: cutoff 36 - 16 = 20 retires exactly the two
/// out-of-window sliding blocks.
///
/// The sibling gate `gemma4_coordinator_conforms_to_the_facade_call_order`
/// runs the same call order through the stricter `NoSettleInCycle`,
/// where the identical settle is refused — a deliberate split between
/// the two checkers.
///
/// Mutations this catches: (a) the coordinator declaring
/// `settle_captures_durable_state() == true`, which would make the
/// permissive checker refuse the lawful settle; (b) the checker admitting
/// the cursor-basis settle; (c) the admitted settle not reaching the
/// coordinator's prune.
#[cfg(target_os = "macos")]
#[test]
fn gemma4_coordinator_is_lawful_under_the_permissive_checker() {
    use crate::engine::spec_paged::NoDurableSettleInCycle;

    let Some(mut coordinator) = maybe_grouped_coordinator(64, 16) else {
        return;
    };
    let sliding_group = sliding_group_id(&coordinator);
    coordinator.reset_scheduled_request(7).expect("reset");
    coordinator
        .record_tokens_all(7, &(0..36).collect::<Vec<_>>())
        .expect("seed the committed prompt");
    assert!(
        !PruneOnlySpecPagedCache::new(&mut coordinator).settle_captures_durable_state(),
        "the coordinator-scope settle is pending-write eval plus a committed-basis \
         prune; the cold-rung walk lives one level up, out of this facade's reach"
    );
    let mut cache = NoDurableSettleInCycle::new(PruneOnlySpecPagedCache::new(&mut coordinator));

    assert!(
        cache.reserve_lookahead(7, 5).expect("reserve"),
        "5 lookahead rows fit every group"
    );
    let ticket = cache
        .record_verify(7, &[100, 101, 102, 103, 104])
        .expect("record anchor + 4 drafts");
    let recorded = allocated_per_group(cache.inner().coordinator());

    // Write-cursor basis: refused, and it must not reach the coordinator.
    let err = cache
        .settle_committed(7, 41)
        .expect_err("a cursor-basis settle inside a cycle must trip");
    assert!(err.contains("L-SETTLE"), "unexpected error text: {err}");
    assert_eq!(
        allocated_per_group(cache.inner().coordinator()),
        recorded,
        "the refused settle must never reach the coordinator"
    );

    // Committed basis: admitted, and it reaches the committed-cutoff prune.
    cache
        .settle_committed(7, 36)
        .expect("a committed-basis settle is lawful inside an open cycle");
    let settled = allocated_per_group(cache.inner().coordinator());
    for (group_id, (before, after)) in recorded.iter().zip(&settled).enumerate() {
        let expected = if group_id == sliding_group { 2 } else { 0 };
        assert_eq!(
            before - after,
            expected,
            "group {group_id}: the admitted settle must retire exactly the \
             out-of-window blocks (36 - 16 = 20)"
        );
    }

    cache
        .commit_cycle(7, ticket, 1)
        .expect("keep the anchor, roll back the 4 rejected drafts");
    assert_eq!(
        cache.frontier(7),
        Some(SpecFrontier {
            attn_tokens: 37,
            recurrent_tokens: None
        }),
        "the facade frontier must land on the committed row count"
    );

    let cache = cache.into_inner();
    let ids: Vec<u32> = cache
        .coordinator()
        .adapter(sliding_group)
        .expect("sliding adapter")
        .block_table_for(7)
        .expect("sliding block table")
        .blocks()
        .iter()
        .map(|block| block.block_id)
        .collect();
    assert_eq!(ids[0], ids[1], "retired blocks share the null sentinel");
    assert!(
        ids[2..].iter().all(|&id| id != ids[0]),
        "the live window the commit returned into must remain physically backed"
    );
}

/// Cross-module gate (adapter ↔ gemma4, T-D0.1 ↔ T-D0.3): the
/// coordinator's committed settle must route through the ADAPTER's
/// committed-cutoff prune — the rollback-range read-back gate
/// (`prune_committed_cutoff_never_nulls_rollback_range`) re-run at
/// coordinator level. The settle here fires between the verify write and
/// the commit ON PURPOSE: that gap is what the committed basis exists
/// for (the family layer loop settles per chunk while speculative rows
/// are pending), and exactly where a cursor-basis prune retires a block
/// the commit's rollback returns the window into.
///
/// Geometry (window 16, block 8): committed 36 → cutoff 20 retires
/// sliding blocks 0-1; a cursor-basis cutoff (41 - 16 = 25) would also
/// null block 2, which holds live positions 21-23 of the post-rollback
/// window [21, 37).
///
/// Mutation this catches: the coordinator (`settle_committed` /
/// `prune_sliding_all_committed`) calling the cursor-basis prune.
#[cfg(target_os = "macos")]
#[test]
fn settle_committed_never_nulls_the_rollback_range_at_coordinator_level() {
    let Some(mut coordinator) = maybe_grouped_coordinator(64, 16) else {
        return;
    };
    let sliding_group = sliding_group_id(&coordinator);
    coordinator.reset_scheduled_request(7).expect("reset");
    coordinator
        .record_tokens_all(7, &(0..36).collect::<Vec<_>>())
        .expect("seed the committed prompt");

    let mut cache = PruneOnlySpecPagedCache::new(&mut coordinator);
    assert!(
        cache.reserve_lookahead(7, 5).expect("reserve"),
        "5 lookahead rows fit every group"
    );
    let ticket = cache
        .record_verify(7, &[100, 101, 102, 103, 104])
        .expect("record anchor + 4 drafts");

    let recorded = allocated_per_group(cache.coordinator());
    cache
        .settle_committed(7, 36)
        .expect("settle at the pre-verify committed frontier");
    let settled = allocated_per_group(cache.coordinator());
    for (group_id, (before, after)) in recorded.iter().zip(&settled).enumerate() {
        let expected = if group_id == sliding_group { 2 } else { 0 };
        assert_eq!(
            before - after,
            expected,
            "group {group_id}: the committed cutoff (36 - 16 = 20) retires \
             blocks 0-1 only — a cursor cutoff (41 - 16 = 25) would take 3"
        );
    }

    cache
        .commit_cycle(7, ticket, 1)
        .expect("roll back the 4 rejected drafts");
    assert_eq!(coordinator.request_token_count_all(7).expect("count"), 37);

    // The post-rollback live window [21, 37) reads back...
    coordinator
        .adapter_mut(sliding_group)
        .expect("sliding adapter")
        .read_kv_range(0, 21, 16)
        .expect("the post-rollback live window must be readable");
    // ...and is physically backed. `read_kv_range`'s liveness floor is
    // cursor-derived and cannot see a null placeholder behind an
    // in-window position, so the Ok alone does not prove backing — the
    // block ids are the proof.
    let ids: Vec<u32> = coordinator
        .adapter(sliding_group)
        .expect("sliding adapter")
        .block_table_for(7)
        .expect("sliding block table")
        .blocks()
        .iter()
        .map(|block| block.block_id)
        .collect();
    assert_eq!(
        ids[0], ids[1],
        "blocks wholly out of the committed window share the null sentinel"
    );
    assert!(
        ids[2..5].iter().all(|&id| id != ids[0]),
        "the rollback range the commit returned into must remain physically backed"
    );
}

/// The facade checks the ticket's sequence FIRST, before the full-accept
/// path can return: a commit that rolls back nothing still names a
/// sequence, and committing sequence A's cycle against sequence B must
/// not pass just because the arithmetic happens to work out.
///
/// Both sequences are parked on the SAME frontier here on purpose, so the
/// ticket's basis-plus-width check and the landing post-condition both
/// pass against the wrong sequence — the sequence id is the only thing
/// left that can catch it. The full accept (`keep == rows`) is the shape
/// that has no other reason to touch the cache at all.
///
/// Mutation this catches: moving the ticket's sequence check after the
/// zero-rollback path in `engine::spec_paged::SpecPagedCache::commit_cycle`
/// — the foreign commit then returns `Ok`.
#[test]
fn a_foreign_ticket_is_refused_on_the_full_accept_path() {
    let Some(mut coordinator) = maybe_grouped_coordinator(64, 16) else {
        return;
    };
    coordinator.reset_scheduled_request(7).expect("reset 7");
    coordinator.reset_scheduled_request(8).expect("reset 8");
    coordinator
        .record_tokens_all(7, &(0..36).collect::<Vec<_>>())
        .expect("seed sequence 7");
    // Sequence 8 sits exactly where sequence 7's cycle will END, so the
    // ticket describes 8's cursor just as well as 7's.
    coordinator
        .record_tokens_all(8, &(0..39).collect::<Vec<_>>())
        .expect("seed sequence 8 at the frontier 7's cycle lands on");

    let mut cache = PruneOnlySpecPagedCache::new(&mut coordinator);
    let ticket = cache
        .record_verify(7, &[100, 101, 102])
        .expect("open sequence 7's cycle");
    assert_eq!(ticket.pre_attn_tokens(), 36);
    assert_eq!(cache.frontier(7).expect("frontier 7").attn_tokens, 39);
    assert_eq!(cache.frontier(8).expect("frontier 8").attn_tokens, 39);

    let err = cache
        .commit_cycle(8, ticket, 3)
        .expect_err("sequence 8 may not be committed with sequence 7's ticket");
    assert!(
        err.contains("sequence 7's verify ticket"),
        "the sequence mismatch must be what is reported, got: {err}"
    );
    assert_eq!(
        cache.frontier(7).expect("frontier 7").attn_tokens,
        39,
        "the refused commit must leave sequence 7's open cycle untouched"
    );
    assert_eq!(
        cache.frontier(8).expect("frontier 8").attn_tokens,
        39,
        "the refused commit must not roll back the sequence it named"
    );
}

/// Bit-equal row `T-1`: the all-rows projection's last row must be the
/// last-only projection exactly. Mutation this catches: a projection row
/// off-by-one in either mode.
#[test]
fn paged_all_rows_logits_row_minus_one_matches_last_only() {
    let config = tiny_paged_config();
    let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)
        .expect("tiny embedding");
    let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))
        .expect("tiny final norm");
    let hidden = MxArray::random_uniform(&[1, 4, config.hidden_size as i64], -1.0, 1.0, None)
        .expect("random hidden");

    let all_rows = project_paged_hidden_rows(
        &hidden,
        &final_norm,
        &embedding,
        &None,
        None,
        &config,
        false,
    )
    .expect("all-rows projection");
    assert_eq!(
        all_rows.shape().unwrap().to_vec(),
        vec![1, 4, config.vocab_size as i64],
        "all-rows mode must keep one logit row per position"
    );
    let last_only =
        project_paged_hidden_rows(&hidden, &final_norm, &embedding, &None, None, &config, true)
            .expect("last-only projection");
    assert_eq!(
        last_only.shape().unwrap().to_vec(),
        vec![config.vocab_size as i64],
        "last-only mode must squeeze to [vocab]"
    );

    let row_t_minus_1 = all_rows
        .slice_axis(1, 3, 4)
        .and_then(|row| row.squeeze(Some(&[0, 1])))
        .expect("slice row T-1");
    assert_bitwise_eq(&row_t_minus_1, &last_only, "all-rows row T-1 vs last-only");

    // The rows must be genuinely per-position, not one row broadcast.
    let row_0 = all_rows
        .slice_axis(1, 0, 1)
        .and_then(|row| row.squeeze(Some(&[0, 1])))
        .expect("slice row 0");
    row_0.eval();
    assert_ne!(
        row_0.to_float32().unwrap().to_vec(),
        last_only.to_float32().unwrap().to_vec(),
        "distinct positions must project to distinct logit rows"
    );
}

/// One sequence's full settle trace: per-settle block-id snapshots for
/// every group, the cursor at each settle, and the captured cold rungs.
struct SettleTrace {
    step_block_ids: Vec<Vec<Vec<u32>>>,
    cursors: Vec<u32>,
    rungs: Vec<(u32, Vec<u32>)>,
}

fn drive_settles(inner: &mut Gemma4Inner, seq_id: u32, committed_basis: bool) -> SettleTrace {
    inner
        .kv_cache_coordinator
        .as_mut()
        .expect("coordinator")
        .reset_scheduled_request(seq_id)
        .expect("reset");
    // Chunks of 3 keep the prune cutoff off block alignment; the 5-chunk
    // lands one settle exactly on the first cold anchor rung
    // (block_size * 4 = 32) so the rung walk actually captures.
    let schedule = [3u32, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3];
    let mut next_token = 0u32;
    let mut step_block_ids = Vec::new();
    let mut cursors = Vec::new();
    for chunk_len in schedule {
        let tokens: Vec<u32> = (next_token..next_token + chunk_len).collect();
        next_token += chunk_len;
        inner
            .kv_cache_coordinator
            .as_mut()
            .expect("coordinator")
            .record_tokens_all(seq_id, &tokens)
            .expect("record chunk");
        let cursor = inner
            .kv_cache_coordinator
            .as_ref()
            .expect("coordinator")
            .full_adapter()
            .current_token_count_for(seq_id)
            .expect("cursor");
        if committed_basis {
            inner
                .settle_grouped_kv_step_at(seq_id, cursor)
                .expect("committed-basis settle");
        } else {
            inner
                .settle_grouped_kv_step(seq_id)
                .expect("cursor-basis settle");
        }
        cursors.push(cursor);
        let coordinator = inner.kv_cache_coordinator.as_ref().expect("coordinator");
        let groups = (0..group_count(coordinator))
            .map(|group_id| {
                coordinator
                    .adapter(group_id)
                    .expect("group adapter")
                    .block_table_for(seq_id)
                    .expect("block table")
                    .blocks()
                    .iter()
                    .map(|block| block.block_id)
                    .collect::<Vec<u32>>()
            })
            .collect::<Vec<_>>();
        step_block_ids.push(groups);
    }
    let rungs = inner
        .grouped_sliding_cold_checkpoints
        .get(&seq_id)
        .map(|checkpoints| {
            checkpoints
                .iter()
                .map(|checkpoint| (checkpoint.boundary, checkpoint.tokens.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    inner
        .kv_cache_coordinator
        .as_mut()
        .expect("coordinator")
        .release_request_all(seq_id)
        .expect("release");
    SettleTrace {
        step_block_ids,
        cursors,
        rungs,
    }
}

/// Per-step nulled-position patterns for one trace: a position is nulled
/// iff its block id equals the trace-final id of position 0 (the shared
/// null sentinel once the window has passed block 0; the sentinel block
/// is adapter-owned and never freed, so no live block can alias it).
fn null_patterns(trace: &SettleTrace) -> Vec<Vec<Vec<bool>>> {
    let final_step = trace.step_block_ids.last().expect("at least one settle");
    let null_ids: Vec<u32> = final_step.iter().map(|ids| ids[0]).collect();
    trace
        .step_block_ids
        .iter()
        .map(|groups| {
            groups
                .iter()
                .zip(&null_ids)
                .map(|(ids, &null_id)| ids.iter().map(|&id| id == null_id).collect())
                .collect()
        })
        .collect()
}

/// When committed == cursor (every autoregressive step), the
/// committed-basis settle must be indistinguishable from the cursor-basis
/// settle: bit-equal nulled-block sets in every group at every step, and
/// identical captured checkpoint rungs. Mutations this catches: the
/// settle refactor changing autoregressive behavior (either basis arm
/// drifting by even one token moves a retirement boundary or drops the
/// on-frontier rung).
#[test]
fn settle_at_committed_equals_cursor_when_equal() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping settle_at_committed_equals_cursor_when_equal: Metal unavailable");
        return;
    }
    let mut inner =
        Gemma4Inner::new(tiny_paged_config()).expect("tiny paged Gemma4Inner must construct");
    if inner.kv_cache_coordinator.is_none() {
        eprintln!("skipping settle_at_committed_equals_cursor_when_equal: no paged coordinator");
        return;
    }

    // A real sliding cold tier so the rung walk actually captures — an
    // empty-ladder run would compare nothing.
    let root = std::env::temp_dir().join(format!(
        "mlx-gemma4-spec-paged-settle-{}",
        std::process::id()
    ));
    let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.clone())
        .expect("temp-dir cold cache must open");
    let policy = sliding_sidecar::policy(&inner.config)
        .expect("tiny geometry must yield a sliding sidecar policy");
    inner
        .kv_cache_coordinator
        .as_mut()
        .expect("coordinator")
        .full_adapter_mut()
        .set_cold_tier(ColdTierContext {
            manager: Arc::new(manager),
            fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                b"gemma4-spec-paged-settle".as_slice(),
            ]),
            sidecar_policy: Some(policy),
        });

    let cursor_trace = drive_settles(&mut inner, 21, false);
    let committed_trace = drive_settles(&mut inner, 22, true);

    assert_eq!(cursor_trace.cursors, committed_trace.cursors);
    assert_eq!(
        null_patterns(&cursor_trace),
        null_patterns(&committed_trace),
        "committed == cursor must null bit-equal block sets in every group at every settle"
    );

    // Non-vacuity: the window actually retired blocks...
    let final_patterns = null_patterns(&cursor_trace);
    let final_step = final_patterns.last().expect("final settle");
    assert!(
        final_step
            .iter()
            .any(|pattern| pattern.iter().filter(|&&nulled| nulled).count() >= 2),
        "the schedule must drive at least two blocks out of a sliding window"
    );
    // ...and the rung walk captured the on-frontier anchor, identically.
    let expected_rungs = vec![(32u32, (0..32).collect::<Vec<u32>>())];
    assert_eq!(
        cursor_trace.rungs, expected_rungs,
        "the cursor-basis settle must capture the rung landing exactly on the frontier"
    );
    assert_eq!(
        committed_trace.rungs, expected_rungs,
        "the committed-basis settle must capture the same rung at committed == cursor"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// The model-level committed-basis settle BELOW the cursor — the gap
/// [`settle_at_committed_equals_cursor_when_equal`] cannot see: with
/// committed < cursor, the rung walk and the sliding prune must both
/// consume the committed frontier, never the write cursor.
///
/// Geometry (window 16, block 8, first anchor rung 32): the cursor lands
/// exactly on the rung, committed 26 sits below it. The committed cutoff
/// (26 - 16 = 10) retires sliding block 0 only; a cursor cutoff
/// (32 - 16 = 16) would also null block 1, which holds live positions
/// 10-15 of the committed window [10, 26). The boundary-32 rows are still
/// live at settle time (`read_sliding_groups_at` succeeds only while the
/// sliding cursor sits exactly on the boundary), so a cursor-basis rung
/// walk WOULD capture a durable checkpoint a rollback below it can no
/// longer retract.
///
/// Mutations this catches:
/// `remember_grouped_sliding_cold_checkpoint_at_frontier` ignoring
/// `committed_tokens` in its frontier (captures boundary 32 > committed);
/// the `Some(committed)` arm of `settle_grouped_kv_step_at_basis` routing
/// to the cursor-basis `prune_sliding_all`.
#[test]
fn settle_below_cursor_stays_on_the_committed_basis() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping settle_below_cursor_stays_on_the_committed_basis: Metal unavailable");
        return;
    }
    let mut inner =
        Gemma4Inner::new(tiny_paged_config()).expect("tiny paged Gemma4Inner must construct");
    if inner.kv_cache_coordinator.is_none() {
        eprintln!(
            "skipping settle_below_cursor_stays_on_the_committed_basis: no paged coordinator"
        );
        return;
    }

    let root = std::env::temp_dir().join(format!(
        "mlx-gemma4-spec-paged-settle-below-{}",
        std::process::id()
    ));
    let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.clone())
        .expect("temp-dir cold cache must open");
    let policy = sliding_sidecar::policy(&inner.config)
        .expect("tiny geometry must yield a sliding sidecar policy");
    inner
        .kv_cache_coordinator
        .as_mut()
        .expect("coordinator")
        .full_adapter_mut()
        .set_cold_tier(ColdTierContext {
            manager: Arc::new(manager),
            fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                b"gemma4-spec-paged-settle-below".as_slice(),
            ]),
            sidecar_policy: Some(policy),
        });
    assert_eq!(
        inner.scheduled_cold_anchor_rungs().first(),
        Some(&32),
        "test precondition: the ladder must publish the rung the cursor lands on"
    );

    let seq_id = 23u32;
    let committed = 26u32;
    {
        let coordinator = inner.kv_cache_coordinator.as_mut().expect("coordinator");
        coordinator.reset_scheduled_request(seq_id).expect("reset");
        coordinator
            .record_tokens_all(seq_id, &(0..32).collect::<Vec<_>>())
            .expect("record 26 committed + 6 speculative rows up to the rung edge");
        assert!(
            coordinator
                .read_sliding_groups_at(seq_id, 32)
                .expect("boundary read")
                .is_some(),
            "test precondition: the boundary-32 rows must be live, so only the \
             committed basis refuses the capture"
        );
    }
    let coordinator = inner.kv_cache_coordinator.as_ref().expect("coordinator");
    let sliding_group = sliding_group_id(coordinator);
    let sliding_block_ids = |inner: &Gemma4Inner| -> Vec<u32> {
        inner
            .kv_cache_coordinator
            .as_ref()
            .expect("coordinator")
            .adapter(sliding_group)
            .expect("sliding adapter")
            .block_table_for(seq_id)
            .expect("sliding block table")
            .blocks()
            .iter()
            .map(|block| block.block_id)
            .collect()
    };
    let ids_before = sliding_block_ids(&inner);
    let allocated_before = allocated_per_group(coordinator);

    inner
        .settle_grouped_kv_step_at(seq_id, committed)
        .expect("committed-basis settle below the cursor");

    let captured: Vec<u32> = inner
        .grouped_sliding_cold_checkpoints
        .get(&seq_id)
        .map(|checkpoints| {
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.boundary)
                .collect()
        })
        .unwrap_or_default();
    assert!(
        captured.iter().all(|&boundary| boundary <= committed),
        "the rung walk must never capture a boundary past the committed frontier, \
         got {captured:?}"
    );

    let allocated_after =
        allocated_per_group(inner.kv_cache_coordinator.as_ref().expect("coordinator"));
    for (group_id, (before, after)) in allocated_before.iter().zip(&allocated_after).enumerate() {
        let expected = if group_id == sliding_group { 1 } else { 0 };
        assert_eq!(
            before - after,
            expected,
            "group {group_id}: the committed cutoff (26 - 16 = 10) retires sliding \
             block 0 only — a cursor cutoff (32 - 16 = 16) would take 2"
        );
    }
    let ids_after = sliding_block_ids(&inner);
    assert_ne!(
        ids_after[0], ids_before[0],
        "block 0 sits wholly below the committed window and must be remapped to \
         the null sentinel"
    );
    assert_eq!(
        ids_after[1..],
        ids_before[1..],
        "every block the committed window [10, 26) still touches must keep its \
         physical backing"
    );

    let coordinator = inner.kv_cache_coordinator.as_mut().expect("coordinator");
    coordinator
        .rollback_last_tokens_all(seq_id, 6)
        .expect("roll back the 6 speculative rows");
    assert_eq!(
        coordinator.request_token_count_all(seq_id).expect("count"),
        committed
    );
    coordinator
        .adapter_mut(sliding_group)
        .expect("sliding adapter")
        .read_kv_range(0, 10, 16)
        .expect("the committed window [10, 26) must be readable after the rollback");
    let null_id = ids_after[0];
    assert!(
        sliding_block_ids(&inner)[1..]
            .iter()
            .all(|&id| id != null_id),
        "the rollback range the settle preserved must remain physically backed"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// One speculative cycle at whichever facade scope is handed in, landing
/// the committed frontier exactly on the first cold anchor rung: 29
/// committed rows, a 5-row verify write to cursor 34, commit keep=3 →
/// committed 32, then the settle under test. The rung is capturable only
/// while the sliding cursor sits exactly on the boundary
/// (`read_sliding_groups_at`), which the rollback is what establishes.
fn drive_cycle_onto_the_cold_rung<C: SpecPagedCache>(cache: &mut C, seq_id: u32) {
    assert!(
        cache.reserve_lookahead(seq_id, 5).expect("reserve"),
        "5 lookahead rows fit every group"
    );
    let ticket = cache
        .record_verify(seq_id, &[200, 201, 202, 203, 204])
        .expect("record anchor + 4 drafts");
    assert_eq!(
        cache.frontier(seq_id).expect("frontier").attn_tokens,
        34,
        "the verify write must sit 5 rows past the committed prompt"
    );
    cache
        .commit_cycle(seq_id, ticket, 3)
        .expect("keep the anchor and 2 drafts");
    assert_eq!(
        cache.frontier(seq_id).expect("frontier").attn_tokens,
        32,
        "the commit must land the committed frontier on the anchor rung"
    );
    cache
        .settle_committed(seq_id, 32)
        .expect("settle at the committed frontier");
}

/// Settle OWNERSHIP: a family-level driver settling at the committed
/// frontier must get the cold-rung walk. [`Gemma4SpecPagedCache`] routes
/// its settle through [`Gemma4Inner::settle_grouped_kv_step_at`] and
/// captures the rung; [`PruneOnlySpecPagedCache`] structurally cannot
/// reach it — its settle is the coordinator's pending-write eval plus the
/// committed prune — so a driver holding the coordinator scope loses rung
/// capture silently. The loss costs acceptance on a warm restore and
/// never correctness, which is why no other gate can see it.
///
/// Both scopes are driven through the SAME call order
/// ([`drive_cycle_onto_the_cold_rung`]), so the facade is the only
/// variable, and the coordinator scope is asserted to still prune — it is
/// lossy, not broken.
///
/// Mutation this catches: routing `Gemma4SpecPagedCache::settle_committed`
/// through the coordinator's prune-only settle — the rung map stays empty.
#[test]
fn family_settle_at_the_committed_frontier_captures_the_cold_rung() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!(
            "skipping family_settle_at_the_committed_frontier_captures_the_cold_rung: \
             Metal unavailable"
        );
        return;
    }
    let mut inner =
        Gemma4Inner::new(tiny_paged_config()).expect("tiny paged Gemma4Inner must construct");
    if inner.kv_cache_coordinator.is_none() {
        eprintln!(
            "skipping family_settle_at_the_committed_frontier_captures_the_cold_rung: \
             no paged coordinator"
        );
        return;
    }

    let root =
        std::env::temp_dir().join(format!("mlx-gemma4-spec-paged-rung-{}", std::process::id()));
    let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.clone())
        .expect("temp-dir cold cache must open");
    let policy = sliding_sidecar::policy(&inner.config)
        .expect("tiny geometry must yield a sliding sidecar policy");
    inner
        .kv_cache_coordinator
        .as_mut()
        .expect("coordinator")
        .full_adapter_mut()
        .set_cold_tier(ColdTierContext {
            manager: Arc::new(manager),
            fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                b"gemma4-spec-paged-rung".as_slice(),
            ]),
            sidecar_policy: Some(policy),
        });
    assert_eq!(
        inner.scheduled_cold_anchor_rungs().first(),
        Some(&32),
        "test precondition: the ladder must publish the rung the commit lands on"
    );

    fn seed(inner: &mut Gemma4Inner, seq_id: u32) {
        let coordinator = inner.kv_cache_coordinator.as_mut().expect("coordinator");
        coordinator
            .reset_scheduled_request(seq_id)
            .expect("reset the sequence");
        coordinator
            .record_tokens_all(seq_id, &(0..29).collect::<Vec<_>>())
            .expect("seed 29 committed rows");
    }

    /// Both settles prune; `ids[0] == ids[1]` is the cutoff (32 - 16 =
    /// 16) having retired sliding blocks 0-1 onto the shared null
    /// sentinel, with the live window still physically backed.
    fn assert_pruned(inner: &Gemma4Inner, sliding_group: usize, seq_id: u32) {
        let ids: Vec<u32> = inner
            .kv_cache_coordinator
            .as_ref()
            .expect("coordinator")
            .adapter(sliding_group)
            .expect("sliding adapter")
            .block_table_for(seq_id)
            .expect("sliding block table")
            .blocks()
            .iter()
            .map(|block| block.block_id)
            .collect();
        assert_eq!(
            ids[0], ids[1],
            "sequence {seq_id}: the settle must retire the two out-of-window \
             sliding blocks onto the null sentinel"
        );
        assert!(
            ids[2..].iter().all(|&id| id != ids[0]),
            "sequence {seq_id}: the committed window must remain physically backed"
        );
    }

    let sliding_group = sliding_group_id(inner.kv_cache_coordinator.as_ref().expect("coordinator"));

    // Family scope: the settle walks the rungs.
    let family_seq = 24u32;
    seed(&mut inner, family_seq);
    drive_cycle_onto_the_cold_rung(&mut Gemma4SpecPagedCache::new(&mut inner), family_seq);
    let captured: Vec<(u32, usize)> = inner
        .grouped_sliding_cold_checkpoints
        .get(&family_seq)
        .map(|checkpoints| {
            checkpoints
                .iter()
                .map(|checkpoint| (checkpoint.boundary, checkpoint.tokens.len()))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        captured,
        vec![(32, 32)],
        "the family settle must capture the rung the committed frontier landed on"
    );
    assert_pruned(&inner, sliding_group, family_seq);
    inner
        .kv_cache_coordinator
        .as_mut()
        .expect("coordinator")
        .release_request_all(family_seq)
        .expect("release the family sequence");

    // Coordinator scope: the same call order prunes exactly as the family
    // settle did and captures nothing, because the rung walk lives one
    // level up.
    let prune_only_seq = 25u32;
    seed(&mut inner, prune_only_seq);
    drive_cycle_onto_the_cold_rung(
        &mut PruneOnlySpecPagedCache::new(
            inner.kv_cache_coordinator.as_mut().expect("coordinator"),
        ),
        prune_only_seq,
    );
    assert!(
        inner
            .grouped_sliding_cold_checkpoints
            .get(&prune_only_seq)
            .is_none_or(|checkpoints| checkpoints.is_empty()),
        "the coordinator scope cannot reach the rung walk"
    );
    assert_pruned(&inner, sliding_group, prune_only_seq);

    let _ = std::fs::remove_dir_all(&root);
}

/// The other half of settle ownership: the family settle can capture a
/// cold checkpoint, which no rollback retracts, so it must DECLARE
/// itself durable and L-SETTLE must keep it out of an open cycle
/// entirely. The coordinator scope is admitted for the identical
/// in-cycle call (`gemma4_coordinator_is_lawful_under_the_permissive_checker`)
/// — the split between the two verdicts is the whole reason the
/// declaration exists.
///
/// Geometry (window 16, block 8): committed 36, verify anchor + 4 drafts
/// → cursor 41, commit keep=1 → 37, whose post-commit cutoff (37 - 16 =
/// 21) retires exactly the two out-of-window sliding blocks.
///
/// Mutation this catches: `Gemma4SpecPagedCache::settle_captures_durable_state`
/// answering `false` — the permissive checker then admits an in-cycle
/// family settle, which is where a rung capture lands on a frontier the
/// commit can still roll back under.
#[test]
fn the_family_settle_is_refused_inside_an_open_cycle() {
    use crate::engine::spec_paged::NoDurableSettleInCycle;

    if !crate::engine::persistence::compiled_forward_backend_available() {
        eprintln!("skipping the_family_settle_is_refused_inside_an_open_cycle: Metal unavailable");
        return;
    }
    let mut inner =
        Gemma4Inner::new(tiny_paged_config()).expect("tiny paged Gemma4Inner must construct");
    if inner.kv_cache_coordinator.is_none() {
        eprintln!(
            "skipping the_family_settle_is_refused_inside_an_open_cycle: no paged coordinator"
        );
        return;
    }

    let seq_id = 27u32;
    {
        let coordinator = inner.kv_cache_coordinator.as_mut().expect("coordinator");
        coordinator
            .reset_scheduled_request(seq_id)
            .expect("reset the sequence");
        coordinator
            .record_tokens_all(seq_id, &(0..36).collect::<Vec<_>>())
            .expect("seed the committed prompt");
    }
    let sliding_group = sliding_group_id(inner.kv_cache_coordinator.as_ref().expect("coordinator"));

    let mut cache = NoDurableSettleInCycle::new(Gemma4SpecPagedCache::new(&mut inner));
    assert!(
        cache.reserve_lookahead(seq_id, 5).expect("reserve"),
        "5 lookahead rows fit every group"
    );
    let ticket = cache
        .record_verify(seq_id, &[100, 101, 102, 103, 104])
        .expect("record anchor + 4 drafts");
    let recorded = allocated_per_group(
        cache
            .inner()
            .model()
            .kv_cache_coordinator
            .as_ref()
            .expect("coordinator"),
    );

    // The committed basis is not enough at family scope: a durable
    // capture may not run inside a cycle at any basis.
    let err = cache
        .settle_committed(seq_id, 36)
        .expect_err("an in-cycle family settle must trip the durability check");
    assert!(
        err.contains("captures durable state"),
        "unexpected error text: {err}"
    );
    assert_eq!(
        allocated_per_group(
            cache
                .inner()
                .model()
                .kv_cache_coordinator
                .as_ref()
                .expect("coordinator")
        ),
        recorded,
        "the refused settle must never reach the coordinator"
    );

    cache
        .commit_cycle(seq_id, ticket, 1)
        .expect("keep the anchor, roll back the 4 rejected drafts");
    cache
        .settle_committed(seq_id, 37)
        .expect("post-commit the family settle is lawful");
    let settled = allocated_per_group(
        cache
            .inner()
            .model()
            .kv_cache_coordinator
            .as_ref()
            .expect("coordinator"),
    );
    for (group_id, (before, after)) in recorded.iter().zip(&settled).enumerate() {
        let expected = if group_id == sliding_group { 2 } else { 0 };
        assert_eq!(
            before - after,
            expected,
            "group {group_id}: the post-commit family settle must retire exactly \
             the out-of-window blocks (37 - 16 = 21)"
        );
    }
}

/// Tap purity on the REAL paged layer loops (text and VLM), driven on a
/// real checkpoint: threading a `DsparkTap` must leave the residual
/// stream bit-identical to a tap-less run while capturing one
/// `[1, T, hidden]` per tapped layer. Mutation this catches: the tap
/// perturbing the forward.
///
/// Env-gated (the paged K/V write path requires bf16 weights, which the
/// random-init tiny model cannot provide):
/// `MLX_TEST_GEMMA4_MODEL_PATH` — a bf16 Gemma4 checkout (the
/// `gemma4_dspark.rs` target). Unset -> skip with a message.
#[test]
fn paged_layer_loop_tap_purity_is_bit_identical() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH") else {
        eprintln!(
            "skipping paged_layer_loop_tap_purity_is_bit_identical: set \
             MLX_TEST_GEMMA4_MODEL_PATH (bf16 Gemma4 checkout)"
        );
        return;
    };
    let (mut inner, _weight_bytes) =
        Gemma4Inner::load_from_dir(&model_path, None).expect("gemma4 checkout must load");
    assert!(
        inner.kv_cache_coordinator.is_some(),
        "a bf16 Gemma4 checkout must build its paged coordinator"
    );
    let layer_kinds = inner.compute_layer_kinds().expect("layer kinds");
    let tokens = [3u32, 9, 17, 25, 33, 41];
    let layer_ids = [0usize, inner.layers.len() / 2, inner.layers.len() - 1];

    let fresh_chunk = |inner: &mut Gemma4Inner, seq_id: u32| {
        let coordinator = inner.kv_cache_coordinator.as_mut().expect("coordinator");
        coordinator.reset_scheduled_request(seq_id).expect("reset");
        coordinator
            .record_tokens_all(seq_id, &tokens)
            .expect("record chunk");
    };

    // Text loop: pass A tap-less, pass B tapped, separate sequences of
    // the same loaded model (fresh chunks at position 0 attend only to
    // in-chunk K/V, so the two passes see identical inputs).
    fresh_chunk(&mut inner, 11);
    let hidden_a = inner
        .run_paged_prefill_layer_loop(&tokens, 0, 0, &layer_kinds, None)
        .expect("tap-less paged loop");
    hidden_a.eval();

    fresh_chunk(&mut inner, 12);
    let mut tap = DsparkTap::new(&layer_ids);
    let hidden_b = inner
        .run_paged_prefill_layer_loop(&tokens, 0, 0, &layer_kinds, Some(&mut tap))
        .expect("tapped paged loop");
    hidden_b.eval();

    assert_bitwise_eq(&hidden_a, &hidden_b, "paged text loop hidden");
    assert_eq!(tap.captured.len(), layer_ids.len());
    let hidden_size = inner.config.hidden_size as i64;
    for capture in &tap.captured {
        assert_eq!(
            capture.shape().unwrap().to_vec(),
            vec![1, tokens.len() as i64, hidden_size]
        );
    }
    let first = tap.captured[0].to_float32().unwrap().to_vec();
    let second = tap.captured[1].to_float32().unwrap().to_vec();
    assert_ne!(first, second, "captures must differ across layers");

    // Unsorted / out-of-range tap ids are rejected before any compute.
    for bad in [vec![2usize, 0], vec![1, 1], vec![inner.layers.len()]] {
        let mut bad_tap = DsparkTap::new(&bad);
        fresh_chunk(&mut inner, 13);
        assert!(
            inner
                .run_paged_prefill_layer_loop(&tokens, 0, 0, &layer_kinds, Some(&mut bad_tap))
                .is_err(),
            "tap layer_ids {bad:?} must be rejected"
        );
    }

    // VLM loop: identical A/B purity over caller-provided embeddings.
    let ids = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64]).expect("ids");
    let embeds = inner
        .embed_tokens
        .forward(&ids)
        .and_then(|embeds| embeds.mul_scalar((inner.config.hidden_size as f64).sqrt()))
        .expect("chunk embeds");

    fresh_chunk(&mut inner, 14);
    let vlm_a = inner
        .run_paged_vlm_prefill_layer_loop(&tokens, &embeds, 0, 0, &layer_kinds, None, None)
        .expect("tap-less paged VLM loop");
    vlm_a.eval();

    fresh_chunk(&mut inner, 15);
    let mut vlm_tap = DsparkTap::new(&layer_ids);
    let vlm_b = inner
        .run_paged_vlm_prefill_layer_loop(
            &tokens,
            &embeds,
            0,
            0,
            &layer_kinds,
            None,
            Some(&mut vlm_tap),
        )
        .expect("tapped paged VLM loop");
    vlm_b.eval();

    assert_bitwise_eq(&vlm_a, &vlm_b, "paged VLM loop hidden");
    assert_eq!(vlm_tap.captured.len(), layer_ids.len());
    for capture in &vlm_tap.captured {
        assert_eq!(
            capture.shape().unwrap().to_vec(),
            vec![1, tokens.len() as i64, hidden_size]
        );
    }
}

/// The rung-candidate walk is inclusive at the frontier and identical
/// for the two bases when committed == cursor — the pure half of the
/// settle-equality gate. Mutation this catches: an exclusive (`<`)
/// frontier comparison dropping the rung a settle lands on exactly.
#[test]
fn cold_rung_candidates_include_the_on_frontier_anchor() {
    let anchors = [32u32, 128, 512];
    assert_eq!(gemma4_cold_rung_candidates(&anchors, 31), Vec::<u32>::new());
    assert_eq!(
        gemma4_cold_rung_candidates(&anchors, 32),
        vec![32],
        "a rung landing exactly on the frontier is capturable"
    );
    assert_eq!(gemma4_cold_rung_candidates(&anchors, 200), vec![32, 128]);
    assert_eq!(
        gemma4_cold_rung_candidates(&anchors, 512),
        vec![32, 128, 512]
    );
}
