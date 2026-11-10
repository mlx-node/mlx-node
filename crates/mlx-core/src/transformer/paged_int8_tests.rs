//! An int8 pool behind the adapter: per-generation scale views, bf16 io,
//! every write entry against the Splash `splash_store_kv_row` math, and
//! every read entry refusing until int8 reads exist.

use std::sync::{Arc, Mutex};

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig};

use super::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow};
use crate::array::{DType, MxArray};

const HEADS: usize = 2;
const QUERY_HEADS: usize = 4;
const DIM: usize = 128;
const BS: usize = 16;
const LAYERS: u32 = 2;
const REFUSAL: &str = "int8 paged reads are not built";

pub(crate) fn pool_config() -> PagedAttentionConfig {
    PagedAttentionConfig {
        block_size: BS as u32,
        gpu_memory_mb: 256,
        head_size: DIM as u32,
        num_kv_heads: HEADS as u32,
        num_layers: LAYERS,
        use_fp8_cache: Some(false),
        max_seq_len: Some(1024),
        max_batch_size: Some(4),
    }
}

/// An adapter over a Shared int8 pool, `None` without Metal.
pub(crate) fn int8_adapter(
    num_blocks: u32,
    max_blocks: u32,
) -> Option<(PagedKVCacheAdapter, Arc<LayerKVPool>)> {
    adapter_with(num_blocks, max_blocks, MetalDtype::Int8)
}

fn adapter_with(
    num_blocks: u32,
    max_blocks: u32,
    dtype: MetalDtype,
) -> Option<(PagedKVCacheAdapter, Arc<LayerKVPool>)> {
    let pool = match LayerKVPool::new(pool_config(), num_blocks, max_blocks, dtype) {
        Ok(pool) => Arc::new(pool),
        Err(error) => {
            eprintln!("skipping int8 adapter test: {error}");
            return None;
        }
    };
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(
        num_blocks, max_blocks, BS as u32,
    )));
    let adapter = PagedKVCacheAdapter::new(allocator, pool.clone(), BS as u32).expect("adapter");
    Some((adapter, pool))
}

/// `[tokens, HEADS, DIM]` bf16 rows and their f32 values. Token 1 head 0 is
/// a zero row; every row has exact `.5` quotients somewhere.
fn rows(tokens: usize, seed: u32) -> (MxArray, Vec<f32>) {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut values = Vec::with_capacity(tokens * HEADS * DIM);
    for token in 0..tokens {
        for head in 0..HEADS {
            for _ in 0..DIM {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((state >> 8) % 511) as f32 / 2.0 - 127.5;
                values.push(if token == 1 && head == 0 { 0.0 } else { value });
            }
        }
    }
    let shape = [tokens as i64, HEADS as i64, DIM as i64];
    let array = MxArray::from_float32(&values, &shape)
        .and_then(|array| array.astype(DType::BFloat16))
        .expect("bf16 rows");
    let exact = array
        .astype(DType::Float32)
        .and_then(|array| array.to_float32())
        .expect("rows back to f32")
        .to_vec();
    (array, exact)
}

/// Splash `splash_store_kv_row`: `clamp(rint(v * 127 / max), ±127)`,
/// scale `max / 127`, a zero row gives 0 / 0.
fn quantize(row: &[f32]) -> (Vec<i8>, f32) {
    let max = row.iter().fold(0.0f32, |max, value| max.max(value.abs()));
    if max == 0.0 {
        return (vec![0; row.len()], 0.0);
    }
    let q = row
        .iter()
        .map(|value| {
            ((value * 127.0f32) / max)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8
        })
        .collect();
    (q, max / 127.0f32)
}

/// Physical slot of `token` in `seq`'s table.
fn slot_of(adapter: &mut PagedKVCacheAdapter, seq: u32, token: u32) -> usize {
    adapter.activate_request(seq).expect("activate");
    let slot = adapter
        .block_table()
        .and_then(|table| table.absolute_slot_index(token))
        .expect("allocated slot");
    usize::try_from(slot).expect("slot")
}

/// Every byte of `values` (`[n, HEADS, DIM]`) at `slots[i]` in `layer`:
/// K at the x = 16 index, V dimension-major, scales at
/// `(block·H + head)·bs + offset` (Splash `splash_q8_scale_index`).
fn assert_stored(pool: &LayerKVPool, layer: usize, slots: &[usize], values: &[f32], what: &str) {
    let data = HEADS * DIM * BS;
    for (row, &slot) in slots.iter().enumerate() {
        let (block, offset) = (slot / BS, slot % BS);
        let blobs = pool
            .read_block_all_layers(block as u32)
            .expect("read block");
        let (key_blob, value_blob) = &blobs[layer];
        assert_eq!(key_blob.len(), data + HEADS * BS * 4);
        for head in 0..HEADS {
            let start = (row * HEADS + head) * DIM;
            let source = &values[start..start + DIM];
            let (q, scale) = quantize(source);
            let scale_at = data + (head * BS + offset) * 4;
            for (blob, side) in [(key_blob, "K"), (value_blob, "V")] {
                let stored = f32::from_le_bytes(blob[scale_at..scale_at + 4].try_into().unwrap());
                assert_eq!(
                    stored.to_bits(),
                    scale.to_bits(),
                    "{what}: {side} scale row {row} head {head}"
                );
            }
            for (d, &expected) in q.iter().enumerate() {
                let key_at = ((head * (DIM / 16) + d / 16) * BS + offset) * 16 + d % 16;
                let value_at = (head * DIM + d) * BS + offset;
                assert_eq!(
                    key_blob[key_at] as i8, expected,
                    "{what}: K row {row} head {head} d {d}"
                );
                assert_eq!(
                    value_blob[value_at] as i8, expected,
                    "{what}: V row {row} head {head} d {d}"
                );
            }
        }
    }
}

fn bf16_zeros(shape: &[i64]) -> MxArray {
    MxArray::zeros(shape, Some(DType::BFloat16)).expect("zeros")
}

/// Mutation M11: `k_scale_array` returns a fresh plane view, which carries
/// no edge to the lazy write and reads stale zeros.
#[test]
fn int8_scale_views_carry_the_write() {
    let Some((mut adapter, pool)) = int8_adapter(4, 8) else {
        return;
    };
    adapter.reset_for_new_request(1).unwrap();
    adapter.allocate_suffix_blocks(20).unwrap();
    adapter
        .record_tokens(&(0..20).collect::<Vec<u32>>())
        .unwrap();
    let (keys, key_values) = rows(20, 1);
    let (values, value_values) = rows(20, 2);
    adapter
        .update_keys_values_native(0, &keys, &values, 0)
        .unwrap();

    let slots: Vec<usize> = (0..20).map(|t| slot_of(&mut adapter, 1, t)).collect();
    let check = |adapter: &mut PagedKVCacheAdapter, blocks: i64| {
        for (side, source) in [("K", &key_values), ("V", &value_values)] {
            let view = if side == "K" {
                adapter.k_scale_array(0).unwrap()
            } else {
                adapter.v_scale_array(0).unwrap()
            };
            assert_eq!(view.dtype().unwrap(), DType::Float32);
            assert_eq!(
                [
                    view.shape_at(0).unwrap(),
                    view.shape_at(1).unwrap(),
                    view.shape_at(2).unwrap()
                ],
                [blocks, HEADS as i64, BS as i64]
            );
            let scales = view.to_float32().unwrap();
            for (row, &slot) in slots.iter().enumerate() {
                for head in 0..HEADS {
                    let start = (row * HEADS + head) * DIM;
                    let (_, scale) = quantize(&source[start..start + DIM]);
                    let at = ((slot / BS) * HEADS + head) * BS + slot % BS;
                    assert_eq!(
                        scales[at].to_bits(),
                        scale.to_bits(),
                        "{side} scale row {row} head {head} at {blocks} blocks"
                    );
                }
            }
        }
    };
    check(&mut adapter, 4);

    adapter.eval_pending_pool_writes().unwrap();
    assert!(pool.grow_to(8).unwrap());
    check(&mut adapter, 8);

    let Some((mut bf16, _)) = adapter_with(4, 4, MetalDtype::BFloat16) else {
        return;
    };
    for scale in [
        bf16.k_scale_array(0).unwrap(),
        bf16.v_scale_array(1).unwrap(),
    ] {
        assert_eq!(scale.shape_at(0).unwrap(), 1);
        assert_eq!(scale.to_float32().unwrap().to_vec(), vec![1.0]);
    }
}

/// Mutation M12: `io_dtype()` returns the cache dtype for Int8, so the bf16
/// write is refused.
#[test]
fn int8_io_dtype_is_bf16() {
    let Some((mut adapter, _)) = int8_adapter(4, 4) else {
        return;
    };
    adapter.reset_for_new_request(1).unwrap();
    adapter.allocate_suffix_blocks(4).unwrap();
    adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
    let (keys, _) = rows(4, 3);
    adapter
        .update_keys_values_native(0, &keys, &keys, 0)
        .expect("an int8 pool takes bf16 rows");

    let half = keys.astype(DType::Float16).unwrap();
    let error = adapter
        .update_keys_values_native(0, &half, &half, 0)
        .expect_err("an int8 pool refuses f16 rows");
    assert!(error.contains("BFloat16"), "got: {error}");

    let query = bf16_zeros(&[1, QUERY_HEADS as i64, DIM as i64]);
    let error = adapter
        .gather_kv_for_decode_graph(0, &query, 1.0, 1.0)
        .map(drop)
        .expect_err("int8 reads refuse");
    assert!(error.contains(REFUSAL), "got: {error}");
}

/// Mutation M13: `prefill_sdpa_cache_dtype` maps Int8 to BFloat16.
#[test]
fn int8_adapter_entries_write_and_refuse() {
    let Some((mut adapter, pool)) = int8_adapter(8, 8) else {
        return;
    };
    assert_eq!(adapter.prefill_sdpa_cache_dtype(), None);
    assert_eq!(adapter.paged_operand_dtype(), Some(DType::BFloat16));

    adapter.begin_request(1).unwrap();
    adapter.allocate_suffix_blocks(20).unwrap();
    adapter
        .record_tokens(&(0..20).collect::<Vec<u32>>())
        .unwrap();
    let (keys, single) = rows(20, 4);
    adapter
        .update_keys_values_native(0, &keys, &keys, 0)
        .unwrap();
    let single_slots: Vec<usize> = (0..20).map(|t| slot_of(&mut adapter, 1, t)).collect();

    adapter.begin_request(2).unwrap();
    adapter.allocate_suffix_blocks(5).unwrap();
    adapter.record_tokens(&[1, 2, 3, 4, 5]).unwrap();
    adapter.begin_request(3).unwrap();
    adapter.allocate_suffix_blocks(4).unwrap();
    adapter.record_tokens(&[6, 7, 8, 9]).unwrap();
    let ragged_rows = [
        PagedRaggedRow {
            seq_id: 2,
            first_logical_position: 2,
            query_len: 3,
        },
        PagedRaggedRow {
            seq_id: 3,
            first_logical_position: 2,
            query_len: 2,
        },
    ];
    let (keys, ragged) = rows(5, 5);
    adapter
        .update_keys_values_native_ragged(1, &keys, &keys, &ragged_rows)
        .unwrap();
    let ragged_slots: Vec<usize> = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3)]
        .into_iter()
        .map(|(seq, t)| slot_of(&mut adapter, seq, t))
        .collect();

    for seq in [2, 3] {
        adapter.allocate_suffix_blocks_for(seq, 6).unwrap();
        adapter.record_tokens(&[10]).unwrap();
    }
    let (keys, batched) = rows(2, 6);
    adapter
        .update_keys_values_native_batched(0, &keys, &keys, &[(2, 5), (3, 4)])
        .unwrap();
    let batched_slots = vec![slot_of(&mut adapter, 2, 5), slot_of(&mut adapter, 3, 4)];

    adapter.eval_pending_pool_writes().unwrap();
    assert_stored(
        &pool,
        0,
        &single_slots,
        &single,
        "update_keys_values_native",
    );
    assert_stored(
        &pool,
        1,
        &ragged_slots,
        &ragged,
        "update_keys_values_native_ragged",
    );
    assert_stored(
        &pool,
        0,
        &batched_slots,
        &batched,
        "update_keys_values_native_batched",
    );

    adapter.activate_request(1).unwrap();
    let q = |tokens: i64| bf16_zeros(&[tokens, QUERY_HEADS as i64, DIM as i64]);
    let mut refusals: Vec<(&str, Result<(), String>)> = vec![
        (
            "decode graph",
            adapter
                .gather_kv_for_decode_graph(0, &q(1), 1.0, 1.0)
                .map(drop),
        ),
        (
            "decode raw",
            adapter.gather_kv_for_decode(0, &q(1), 1.0, 1.0).map(drop),
        ),
        (
            "prefill chunk",
            adapter
                .gather_kv_for_prefill_chunk(0, &q(4), 16, 1.0)
                .map(drop),
        ),
        (
            "prefill varlen",
            adapter
                .gather_kv_for_prefill_chunk_varlen(0, &q(4), 16, 1.0)
                .map(drop),
        ),
        (
            "prefill sdpa",
            adapter.gather_kv_for_prefill_sdpa(0, 20).map(drop),
        ),
        (
            "dense cache hit",
            adapter
                .gather_kv_for_dense_cache_hit_prefill(0, 20)
                .map(drop),
        ),
        ("read_kv_range", adapter.read_kv_range(0, 0, 20).map(drop)),
    ];
    refusals.push((
        "decode batched",
        adapter
            .gather_kv_for_decode_graph_batched(0, &q(2), &[2, 3], 1.0, 1.0)
            .map(drop),
    ));
    refusals.push((
        "ragged",
        adapter
            .gather_kv_for_ragged_graph(0, &q(5), &ragged_rows, 1.0, 1.0)
            .map(drop),
    ));
    for (entry, result) in refusals {
        let error = result.expect_err(entry);
        assert!(error.contains(REFUSAL), "{entry}: {error}");
    }
}
