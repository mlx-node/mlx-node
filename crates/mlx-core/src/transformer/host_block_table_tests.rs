//! Checks of the varlen block table built on the host
//! against the order that reshaped the compact ids and evaluated them, and
//! against the adapter's own block table: every value the checks refuse,
//! rows added, rolled back to every keep count, a released request replayed
//! on other blocks, and two requests switched under the same call.

use std::sync::Arc;

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{PagedAttentionConfig, PhysicalBlock, SequenceBlockTable};

use crate::transformer::paged_int8_tests::adapter_for;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::paged_metadata_cache::{PagedMetadataCache, RESHAPED_BLOCK_TABLE};

const BLOCK: u32 = 16;

/// What one call hands the attention kernel.
#[derive(Debug, PartialEq)]
struct Inputs {
    table: Vec<i32>,
    shape: Vec<i64>,
    seq_lens: Vec<i32>,
    cu_seqlens_q: Vec<i32>,
    count: u32,
    total: u32,
}

/// One call on a fresh metadata cache, with the reshaped table (`reshaped`)
/// or the host-built one.
fn inputs(
    table: Option<&SequenceBlockTable>,
    committed: u32,
    rows: u32,
    pool_blocks: u32,
    reshaped: bool,
) -> Result<Inputs, String> {
    RESHAPED_BLOCK_TABLE.with(|flag| flag.set(reshaped));
    let mut cache = PagedMetadataCache::default();
    let out = cache.varlen_prefill_attention_inputs(table, committed, rows, BLOCK, pool_blocks);
    RESHAPED_BLOCK_TABLE.with(|flag| flag.set(false));
    let (block_table, seq_lens, cu_seqlens_q, count, total) = out?;
    Ok(Inputs {
        shape: block_table.shape().map_err(|e| e.to_string())?.to_vec(),
        table: block_table.to_int32().map_err(|e| e.to_string())?.to_vec(),
        seq_lens: seq_lens.to_int32().map_err(|e| e.to_string())?.to_vec(),
        cu_seqlens_q: cu_seqlens_q.to_int32().map_err(|e| e.to_string())?.to_vec(),
        count,
        total,
    })
}

fn table_of(ids: &[u32], tokens: u32) -> SequenceBlockTable {
    let mut table = SequenceBlockTable::new(1, BLOCK);
    for &id in ids {
        table.add_block(Arc::new(PhysicalBlock::new(id)));
    }
    table.set_num_tokens(tokens);
    table
}

/// Every value a check of the varlen metadata refuses is refused by the
/// host-built table with the text the reshaped order gave: no table, no
/// rows, a u32 overflow, a context past i32, rows not recorded, a block id
/// equal to the pool's count, a block id past i32 (negative as int32), a
/// table with fewer blocks than the rows need, a table with no block, and
/// an empty pool.
/// Mutation v-h1-short (the block range is clipped to the table, not
/// refused) turns it red.
#[test]
fn bad_table_values_are_refused_as_before() {
    let good = table_of(&[7, 3, 9, 1, 12], 70);
    let past_i32 = table_of(&[7, 3, u32::MAX, 1, 12], 70);
    let short = table_of(&[7, 3], 70);
    let empty = table_of(&[], 8);
    let cases: [(&str, Option<&SequenceBlockTable>, u32, u32, u32, &str); 10] = [
        ("no table", None, 62, 8, 13, "before reset_for_new_request"),
        ("no rows", Some(&good), 62, 0, 13, "query_len > 0"),
        (
            "u32 overflow",
            Some(&good),
            u32::MAX,
            1,
            13,
            "total context overflow",
        ),
        (
            "past i32",
            Some(&good),
            i32::MAX as u32,
            1,
            13,
            "exceeds int32 range",
        ),
        (
            "rows not recorded",
            Some(&good),
            62,
            9,
            13,
            "recorded token count 70 is less than",
        ),
        (
            "id equal to the pool count",
            Some(&good),
            62,
            8,
            12,
            "block_table[4]=12 out of range for pool block count 12",
        ),
        (
            "id past i32",
            Some(&past_i32),
            62,
            8,
            13,
            "block_table[2]=-1 out of range for pool block count 13",
        ),
        (
            "fewer blocks than the rows need",
            Some(&short),
            62,
            8,
            13,
            "block range exceeds recorded capacity",
        ),
        (
            "no block",
            Some(&empty),
            0,
            8,
            13,
            "block range exceeds recorded capacity",
        ),
        (
            "empty pool",
            Some(&good),
            62,
            8,
            0,
            "block_table[0]=7 out of range for pool block count 0",
        ),
    ];
    for (tag, table, committed, rows, pool, want) in cases {
        let before = inputs(table, committed, rows, pool, true)
            .err()
            .unwrap_or_else(|| panic!("{tag}: the reshaped order took the value"));
        let after = inputs(table, committed, rows, pool, false)
            .err()
            .unwrap_or_else(|| panic!("{tag}: the host table took the value"));
        assert!(before.contains(want), "{tag}: reshaped order said {before}");
        assert_eq!(after, before, "{tag}: the refusal text changed");
    }
    // The same tables with good arguments pass in both orders.
    for (tag, table, committed, rows, pool) in [
        ("whole table", &good, 62u32, 8u32, 13u32),
        ("first block only", &good, 0, 16, 13),
        ("one row into the second block", &good, 16, 1, 8),
        ("short table inside its blocks", &short, 24, 8, 8),
    ] {
        let before = inputs(Some(table), committed, rows, pool, true)
            .unwrap_or_else(|e| panic!("{tag}: reshaped order: {e}"));
        let after = inputs(Some(table), committed, rows, pool, false)
            .unwrap_or_else(|e| panic!("{tag}: host table: {e}"));
        assert_eq!(after, before, "{tag}");
    }
}

fn adapter(blocks: u32) -> Option<PagedKVCacheAdapter> {
    let config = PagedAttentionConfig {
        block_size: BLOCK,
        gpu_memory_mb: 256,
        head_size: 32,
        num_kv_heads: 1,
        num_layers: 1,
        max_seq_len: Some(4096),
        max_batch_size: Some(2),
    };
    adapter_for(config, blocks, blocks, MetalDtype::Int8).map(|(adapter, _)| adapter)
}

/// The table both orders build for the active request's next `rows` rows is
/// the first `ceil(total / 16)` ids of the adapter's block table.
fn check(adapter: &PagedKVCacheAdapter, committed: u32, rows: u32, tag: &str) -> Vec<i32> {
    let ids = adapter
        .block_table()
        .map(|table| table.block_ids())
        .unwrap_or_default();
    let total = committed + rows;
    let count = total.div_ceil(BLOCK);
    let want = Inputs {
        table: ids[..count as usize].iter().map(|&id| id as i32).collect(),
        shape: vec![1, i64::from(count)],
        seq_lens: vec![total as i32],
        cu_seqlens_q: vec![0, rows as i32],
        count,
        total,
    };
    let pool = adapter.layer_kv_pool().num_blocks();
    for reshaped in [true, false] {
        let got = inputs(adapter.block_table(), committed, rows, pool, reshaped)
            .unwrap_or_else(|e| panic!("{tag} (reshaped {reshaped}): {e}"));
        assert_eq!(got, want, "{tag} (reshaped {reshaped})");
    }
    want.table
}

fn tokens(n: u32, salt: u32) -> Vec<u32> {
    (0..n).map(|i| (i * 7 + salt) % 97).collect()
}

/// The host-built table against the adapter's block table through a
/// request's life: prompt chunks, 8-row verify cycles rolled back to every
/// keep count (0 to 8, across block edges), a release, another request on
/// the freed blocks, the first request replayed on other blocks, and the
/// two requests switched under one (committed, rows) call.
/// Mutation v-h1-seqlens (seq_lens holds the committed rows only) turns it
/// red.
#[test]
fn the_host_table_follows_the_request_through_rollback_and_replay() {
    let Some(mut adapter) = adapter(48) else {
        eprintln!(
            "SKIP the_host_table_follows_the_request_through_rollback_and_replay: no adapter"
        );
        return;
    };
    eprintln!(
        "V2_H1 adapter of {} blocks",
        adapter.layer_kv_pool().num_blocks()
    );
    adapter.begin_request(0).expect("adapter call");
    adapter.allocate_suffix_blocks(40).expect("adapter call");
    adapter.record_tokens(&tokens(40, 3)).expect("adapter call");
    check(&adapter, 0, 40, "prompt in one chunk");
    check(&adapter, 24, 16, "prompt's last 16 rows");
    adapter.record_tokens(&tokens(16, 5)).expect("adapter call");
    check(&adapter, 40, 16, "a 16-row chunk");
    adapter.record_tokens(&tokens(1, 6)).expect("adapter call");
    check(&adapter, 56, 1, "a 1-row chunk");

    // Verify cycles: 8 rows booked, `keep` of them kept.
    for keep in 0..=8u32 {
        let committed = adapter.current_token_count();
        adapter
            .record_tokens(&tokens(8, keep))
            .expect("adapter call");
        check(
            &adapter,
            committed,
            8,
            &format!("verify cycle, keep {keep}"),
        );
        adapter
            .rollback_last_tokens(8 - keep)
            .expect("adapter call");
        assert_eq!(adapter.current_token_count(), committed + keep);
        if keep > 0 {
            check(
                &adapter,
                committed,
                keep,
                &format!("kept rows after keep {keep}"),
            );
        }
    }
    let grown = adapter.current_token_count();
    assert_eq!(grown, 57 + 36, "the cycles kept 0 + 1 + .. + 8 rows");
    let first_ids = check(&adapter, grown - 8, 8, "the request before its release");

    // Preemption: the request is released, another takes blocks, and the
    // first is replayed from its tokens.
    let history = adapter.request_tokens().to_vec();
    adapter.release_request().expect("adapter call");
    adapter.begin_request(1).expect("adapter call");
    adapter
        .record_tokens(&tokens(24, 11))
        .expect("adapter call");
    let other_ids = check(&adapter, 0, 24, "the other request");
    adapter.begin_request(0).expect("adapter call");
    adapter.record_tokens(&history).expect("adapter call");
    let replayed_ids = check(&adapter, 0, grown, "the replayed request");
    assert_ne!(
        replayed_ids, first_ids,
        "the replay took the released blocks in their old order: the case shows nothing"
    );
    assert!(
        other_ids.iter().all(|id| !replayed_ids.contains(id)),
        "two live requests share a block"
    );
    check(&adapter, grown - 8, 8, "the replayed request's last rows");

    // Same (committed, rows) for both requests: each gets its own blocks.
    adapter.activate_request(1).expect("adapter call");
    assert_eq!(
        check(&adapter, 16, 8, "request 1 after a switch"),
        other_ids
    );
    adapter.activate_request(0).expect("adapter call");
    assert_eq!(
        check(&adapter, 16, 8, "request 0 after a switch"),
        replayed_ids[..2]
    );
}
