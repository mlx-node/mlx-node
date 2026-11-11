//! An int8 pool behind the adapter: per-generation scale views, bf16 io,
//! every write entry against the Splash `splash_store_kv_row` math, every
//! attention read entry against a bf16 pool, and the dequantizing dense reads
//! against the flat int8 rows.

use std::sync::{Arc, Mutex};

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig};

use super::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow};
use crate::array::kv_int8::Int8KvRows;
use crate::array::{DType, MxArray};

const HEADS: usize = 2;
const QUERY_HEADS: usize = 4;
const DIM: usize = 128;
const BS: usize = 16;
const LAYERS: u32 = 2;

pub(crate) fn pool_config() -> PagedAttentionConfig {
    PagedAttentionConfig {
        block_size: BS as u32,
        gpu_memory_mb: 256,
        head_size: DIM as u32,
        num_kv_heads: HEADS as u32,
        num_layers: LAYERS,
        max_seq_len: Some(1024),
        max_batch_size: Some(4),
    }
}

const D256_HEADS: usize = 4;
const D256_DIM: usize = 256;

fn pool_config_d256() -> PagedAttentionConfig {
    PagedAttentionConfig {
        head_size: D256_DIM as u32,
        num_kv_heads: D256_HEADS as u32,
        ..pool_config()
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
    adapter_for(pool_config(), num_blocks, max_blocks, dtype)
}

fn adapter_for(
    config: PagedAttentionConfig,
    num_blocks: u32,
    max_blocks: u32,
    dtype: MetalDtype,
) -> Option<(PagedKVCacheAdapter, Arc<LayerKVPool>)> {
    let block_size = config.block_size;
    let pool = match LayerKVPool::new(config, num_blocks, max_blocks, dtype) {
        Ok(pool) => Arc::new(pool),
        Err(error) => {
            eprintln!("skipping int8 adapter test: {error}");
            return None;
        }
    };
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(
        num_blocks, max_blocks, block_size,
    )));
    let adapter = PagedKVCacheAdapter::new(allocator, pool.clone(), block_size).expect("adapter");
    Some((adapter, pool))
}

/// `[tokens, HEADS, DIM]` bf16 rows and their f32 values. Token 1 head 0 is
/// a zero row; every row has exact `.5` quotients somewhere.
fn rows(tokens: usize, seed: u32) -> (MxArray, Vec<f32>) {
    shaped_rows(tokens, HEADS, DIM, seed, None)
}

/// [`rows`] over `heads` x `dim`; with a gain, each row times
/// `gain · 2^((3t + h) mod 5)` (powers of two, exact in bf16), so
/// neighbouring rows carry different scales.
fn shaped_rows(
    tokens: usize,
    heads: usize,
    dim: usize,
    seed: u32,
    gain: Option<f32>,
) -> (MxArray, Vec<f32>) {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut values = Vec::with_capacity(tokens * heads * dim);
    for token in 0..tokens {
        for head in 0..heads {
            let row_gain =
                gain.map_or(1.0, |gain| gain * (1u32 << ((3 * token + head) % 5)) as f32);
            for _ in 0..dim {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((state >> 8) % 511) as f32 / 2.0 - 127.5;
                values.push(if token == 1 && head == 0 {
                    0.0
                } else {
                    value * row_gain
                });
            }
        }
    }
    let shape = [tokens as i64, heads as i64, dim as i64];
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
    let output = adapter
        .gather_kv_for_decode_graph(0, &query, 1.0, 1.0)
        .expect("an int8 pool reads with a bf16 query");
    assert_eq!(output.dtype().unwrap(), DType::BFloat16);
    assert_eq!(
        output.shape().unwrap().as_ref(),
        [1, QUERY_HEADS as i64, DIM as i64]
    );
}

/// `[n, H, D]` rows -> `[1, H, n, D]`, the dense readers' layout.
fn heads_major(rows: &MxArray, tokens: usize, heads: usize, dim: usize) -> MxArray {
    rows.reshape(&[1, tokens as i64, heads as i64, dim as i64])
        .and_then(|rows| rows.transpose(Some(&[0, 2, 1, 3])))
        .and_then(|rows| rows.copy())
        .expect("heads-major rows")
}

fn bits(array: &MxArray) -> Vec<u16> {
    assert_eq!(array.dtype().unwrap(), DType::BFloat16);
    array.to_uint16_native().expect("bf16 bits")
}

/// The dense gather (sdpa and dense cache hit) and `read_kv_range` of an int8
/// pool hand back exactly the flat int8 cache's BF16 rows: the flat quantizer
/// then `dequantize_kv_rows`, the same op. Layer 1 rows are 4x layer 0's.
///
/// Mutation M2: the dense gather takes the scale planes `[B, H, bs]` into
/// `[1, H, B·bs]` with no transpose → red. Mutation M3: `read_kv_range`
/// unpacks int8 K with x = 8 → red.
#[test]
fn int8_dense_reads_dequantize_like_the_flat_rows() {
    dense_reads_match_flat(pool_config_d256(), 53, &[(0, 53), (19, 30)]);
}

/// A dense gather inside one page hands the dequantize op V rows whose head
/// dimension has a stride of the block size (the reshape after the page
/// transpose is a view); they still read exactly the flat rows.
///
/// Mutation U1: the dequantize kernel reads the head dimension at stride 1
/// → red (V).
#[test]
fn int8_one_page_dense_reads_dequantize_like_the_flat_rows() {
    for block_size in [16, 32] {
        let config = PagedAttentionConfig {
            block_size,
            ..pool_config_d256()
        };
        dense_reads_match_flat(config, 9, &[(0, 9), (2, 5)]);
    }
}

/// Writes `tokens` rows per layer into a 4-block int8 pool, then checks the
/// dense gathers over `[0, tokens)` and `read_kv_range` over each
/// `(start, count)` bit for bit against the flat int8 rows' dequantize.
fn dense_reads_match_flat(config: PagedAttentionConfig, tokens: usize, ranges: &[(u32, u32)]) {
    let block_size = config.block_size;
    let Some((mut adapter, _)) = adapter_for(config, 4, 4, MetalDtype::Int8) else {
        return;
    };
    adapter.reset_for_new_request(1).unwrap();
    adapter.allocate_suffix_blocks(tokens as u32).unwrap();
    adapter
        .record_tokens(&(0..tokens as u32).collect::<Vec<u32>>())
        .unwrap();
    let mut flat = Vec::new();
    for layer in 0..LAYERS {
        let gain = Some(if layer == 0 { 1.0 } else { 4.0 });
        let (keys, _) = shaped_rows(tokens, D256_HEADS, D256_DIM, 20 + layer, gain);
        let (values, _) = shaped_rows(tokens, D256_HEADS, D256_DIM, 30 + layer, gain);
        adapter
            .update_keys_values_native(layer, &keys, &values, 0)
            .unwrap();
        let rows = Int8KvRows::quantize(
            &heads_major(&keys, tokens, D256_HEADS, D256_DIM),
            &heads_major(&values, tokens, D256_HEADS, D256_DIM),
        )
        .unwrap();
        flat.push(rows);
    }

    for layer in 0..LAYERS {
        let rows = &flat[layer as usize];
        let mut reads: Vec<(String, (MxArray, MxArray), (i64, i64))> = vec![
            (
                "prefill sdpa".to_string(),
                adapter
                    .gather_kv_for_prefill_sdpa(layer, tokens as u32)
                    .unwrap(),
                (0, tokens as i64),
            ),
            (
                "dense cache hit".to_string(),
                adapter
                    .gather_kv_for_dense_cache_hit_prefill(layer, tokens as u32)
                    .map(|(keys, values, _)| (keys, values))
                    .unwrap(),
                (0, tokens as i64),
            ),
        ];
        for &(start, count) in ranges {
            reads.push((
                format!("read_kv_range [{start}, +{count})"),
                adapter.read_kv_range(layer, start, count).unwrap(),
                (start as i64, (start + count) as i64),
            ));
        }
        for (entry, (keys, values), (start, end)) in reads {
            let (want_keys, want_values) =
                rows.slice_tokens(start, end).unwrap().dequantize().unwrap();
            for (side, got, want) in [("K", &keys, &want_keys), ("V", &values, &want_values)] {
                assert_eq!(
                    got.shape().unwrap().as_ref(),
                    want.shape().unwrap().as_ref(),
                    "bs {block_size} layer {layer} {entry} {side} shape"
                );
                let (got, want) = (bits(got), bits(want));
                let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differing, 0,
                    "bs {block_size} layer {layer} {entry} {side}: {differing} bf16 values differ"
                );
            }
        }
    }
}

/// `[tokens, QUERY_HEADS, DIM]` bf16 queries, uniform in ±0.002.
fn queries(tokens: usize, seed: u32) -> MxArray {
    let mut state = seed.wrapping_mul(747_796_405).wrapping_add(7);
    let values: Vec<f32> = (0..tokens * QUERY_HEADS * DIM)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 8) % 4001) as f32 / 1000.0 - 2.0) / 1000.0
        })
        .collect();
    MxArray::from_float32(&values, &[tokens as i64, QUERY_HEADS as i64, DIM as i64])
        .and_then(|array| array.astype(DType::BFloat16))
        .expect("queries")
}

struct EntryWrites {
    single: Vec<f32>,
    ragged: Vec<f32>,
    batched: Vec<f32>,
    single_slots: Vec<usize>,
    ragged_slots: Vec<usize>,
    batched_slots: Vec<usize>,
}

const RAGGED_ROWS: [PagedRaggedRow; 2] = [
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

/// The ragged read over each sequence's last rows after the batched write.
const RAGGED_READ_ROWS: [PagedRaggedRow; 2] = [
    PagedRaggedRow {
        seq_id: 2,
        first_logical_position: 3,
        query_len: 3,
    },
    PagedRaggedRow {
        seq_id: 3,
        first_logical_position: 3,
        query_len: 2,
    },
];

/// The same writes on any pool: sequence 1 (20 tokens) on both layers, then
/// sequences 2 (6 tokens) and 3 (5 tokens) on layer 1 through the single,
/// ragged and batched entries, so every position a read reaches is written.
/// Layer 1 rows are 4x layer 0's.
fn write_entries(adapter: &mut PagedKVCacheAdapter) -> EntryWrites {
    let gain = |layer: u32| Some(if layer == 0 { 1.0 } else { 4.0 });
    adapter.begin_request(1).unwrap();
    adapter.allocate_suffix_blocks(20).unwrap();
    adapter
        .record_tokens(&(0..20).collect::<Vec<u32>>())
        .unwrap();
    let mut single = Vec::new();
    for layer in 0..LAYERS {
        let (keys, exact) = shaped_rows(20, HEADS, DIM, 4 + layer, gain(layer));
        adapter
            .update_keys_values_native(layer, &keys, &keys, 0)
            .unwrap();
        if layer == 0 {
            single = exact;
        }
    }
    let single_slots: Vec<usize> = (0..20).map(|t| slot_of(adapter, 1, t)).collect();

    for (seq, tokens) in [(2u32, [1u32, 2, 3, 4, 5].as_slice()), (3, &[6, 7, 8, 9])] {
        adapter.begin_request(seq).unwrap();
        adapter.allocate_suffix_blocks(tokens.len() as u32).unwrap();
        adapter.record_tokens(&tokens[..2]).unwrap();
        let (keys, _) = shaped_rows(2, HEADS, DIM, 40 + seq, gain(1));
        adapter
            .update_keys_values_native(1, &keys, &keys, 0)
            .unwrap();
        adapter.record_tokens(&tokens[2..]).unwrap();
    }
    let (keys, ragged) = shaped_rows(5, HEADS, DIM, 5, gain(1));
    adapter
        .update_keys_values_native_ragged(1, &keys, &keys, &RAGGED_ROWS)
        .unwrap();
    let ragged_slots: Vec<usize> = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3)]
        .into_iter()
        .map(|(seq, t)| slot_of(adapter, seq, t))
        .collect();

    for seq in [2, 3] {
        adapter.allocate_suffix_blocks_for(seq, 6).unwrap();
        adapter.record_tokens(&[10]).unwrap();
    }
    let (keys, batched) = shaped_rows(2, HEADS, DIM, 6, gain(1));
    adapter
        .update_keys_values_native_batched(1, &keys, &keys, &[(2, 5), (3, 4)])
        .unwrap();
    let batched_slots = vec![slot_of(adapter, 2, 5), slot_of(adapter, 3, 4)];
    adapter.eval_pending_pool_writes().unwrap();
    EntryWrites {
        single,
        ragged,
        batched,
        single_slots,
        ragged_slots,
        batched_slots,
    }
}

/// Every attention read entry on layer 1 of `adapter`, in a fixed order.
fn read_entries(adapter: &mut PagedKVCacheAdapter) -> Vec<(&'static str, MxArray)> {
    let scale = 1.0 / (DIM as f32).sqrt();
    adapter.activate_request(1).unwrap();
    let mut reads = vec![
        (
            "decode graph",
            adapter
                .gather_kv_for_decode_graph(1, &queries(1, 1), scale, 1.0)
                .unwrap(),
        ),
        (
            "decode raw",
            adapter
                .gather_kv_for_decode(1, &queries(1, 1), scale, 1.0)
                .unwrap(),
        ),
        (
            "prefill chunk",
            adapter
                .gather_kv_for_prefill_chunk(1, &queries(4, 2), 16, scale)
                .unwrap(),
        ),
        (
            "prefill varlen",
            adapter
                .gather_kv_for_prefill_chunk_varlen(1, &queries(4, 2), 16, scale)
                .unwrap(),
        ),
    ];
    reads.push((
        "decode batched",
        adapter
            .gather_kv_for_decode_graph_batched(1, &queries(2, 3), &[2, 3], scale, 1.0)
            .unwrap(),
    ));
    reads.push((
        "ragged",
        adapter
            .gather_kv_for_ragged_graph(1, &queries(5, 4), &RAGGED_READ_ROWS, scale, 1.0)
            .unwrap(),
    ));
    reads
}

/// Every write entry stores the Splash `splash_store_kv_row` bytes, and
/// every attention read entry on an int8 pool tracks the same entry on a bf16
/// pool holding the same rows within the int8 noise (a wrong plane is off by
/// the 4x layer gain).
///
/// Mutation M4: `k_scale_array` hands layer 0's plane to a layer 1 read → red.
/// Mutation M13: `prefill_sdpa_cache_dtype` maps Int8 to BFloat16.
#[test]
fn int8_adapter_entries_write_and_read() {
    let Some((mut adapter, pool)) = int8_adapter(8, 8) else {
        return;
    };
    let Some((mut reference, _)) = adapter_with(8, 8, MetalDtype::BFloat16) else {
        return;
    };
    assert_eq!(adapter.prefill_sdpa_cache_dtype(), None);
    assert_eq!(adapter.paged_operand_dtype(), Some(DType::BFloat16));

    let writes = write_entries(&mut adapter);
    write_entries(&mut reference);
    assert_stored(
        &pool,
        0,
        &writes.single_slots,
        &writes.single,
        "update_keys_values_native",
    );
    assert_stored(
        &pool,
        1,
        &writes.ragged_slots,
        &writes.ragged,
        "update_keys_values_native_ragged",
    );
    assert_stored(
        &pool,
        1,
        &writes.batched_slots,
        &writes.batched,
        "update_keys_values_native_batched",
    );

    let int8 = read_entries(&mut adapter);
    let bf16 = read_entries(&mut reference);
    for ((entry, got), (_, want)) in int8.iter().zip(&bf16) {
        assert_eq!(got.dtype().unwrap(), DType::BFloat16, "{entry}");
        assert_eq!(
            got.shape().unwrap().as_ref(),
            want.shape().unwrap().as_ref(),
            "{entry}"
        );
        let got = got
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let want = want
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        assert!(
            got.iter().all(|value| value.is_finite()),
            "{entry}: non-finite output"
        );
        let magnitude = want.iter().fold(0.0f32, |max, value| max.max(value.abs()));
        let (max, sum) = got
            .iter()
            .zip(&want)
            .fold((0.0f32, 0.0f64), |(max, sum), (a, b)| {
                let delta = (a - b).abs();
                (max.max(delta), sum + f64::from(delta))
            });
        let mean = sum / got.len() as f64;
        eprintln!(
            "{entry}: magnitude {magnitude} max |d| {max} ({:.3e}) mean |d| {mean} ({:.3e})",
            max / magnitude,
            mean / f64::from(magnitude)
        );
        assert!(magnitude > 0.0, "{entry}: zero output");
        assert!(
            max <= magnitude / 32.0,
            "{entry}: max |d| {max} > magnitude {magnitude} / 32"
        );
    }
}

/// `[tokens, heads, dim]` bf16 queries of about ±0.002.
fn wide_queries(tokens: usize, heads: usize, dim: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(747_796_405).wrapping_add(7);
    (0..tokens * heads * dim)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 8) % 4001) as f32 / 1000.0 - 2.0) / 1000.0
        })
        .collect()
}

fn bf16_array(values: &[f32], shape: &[i64]) -> MxArray {
    MxArray::from_float32(values, shape)
        .and_then(|array| array.astype(DType::BFloat16))
        .expect("bf16 array")
}

fn host(array: &MxArray) -> Vec<f32> {
    array
        .astype(DType::Float32)
        .and_then(|array| array.to_float32())
        .expect("read back")
        .to_vec()
}

/// One sequence of `tokens` seeded rows in a one-layer int8 pool (D 256,
/// 4 heads, bs 16), written 2,048 rows at a time.
fn long_int8_adapter(tokens: u32) -> Option<PagedKVCacheAdapter> {
    let config = PagedAttentionConfig {
        num_layers: 1,
        max_seq_len: Some(tokens + 2048),
        ..pool_config_d256()
    };
    let blocks = tokens.div_ceil(BS as u32) + 1;
    let (mut adapter, _) = adapter_for(config, blocks, blocks, MetalDtype::Int8)?;
    adapter.begin_request(1).unwrap();
    adapter.allocate_suffix_blocks(tokens).unwrap();
    let mut first = 0u32;
    while first < tokens {
        let rows = (tokens - first).min(2048) as usize;
        adapter
            .record_tokens(
                &(first..first + rows as u32)
                    .map(|t| t % 1000)
                    .collect::<Vec<u32>>(),
            )
            .unwrap();
        let (keys, _) = shaped_rows(rows, D256_HEADS, D256_DIM, 100 + first, None);
        let (values, _) = shaped_rows(rows, D256_HEADS, D256_DIM, 200 + first, None);
        adapter
            .update_keys_values_native(0, &keys, &values, first)
            .unwrap();
        adapter.eval_pending_pool_writes().unwrap();
        first += rows as u32;
    }
    Some(adapter)
}

/// The varlen read of the last `rows` of `context` tokens, and each sampled
/// row read alone: the same bits (a row's partitions and reduce read only
/// that row, however the dispatch tiles the rows).
fn varlen_rows_match_single_rows(
    adapter: &mut PagedKVCacheAdapter,
    context: u32,
    rows: usize,
    sampled: &[usize],
) {
    const QUERY_HEADS_D256: usize = 24;
    let scale = 1.0 / (D256_DIM as f32).sqrt();
    let width = QUERY_HEADS_D256 * D256_DIM;
    let q = wide_queries(rows, QUERY_HEADS_D256, D256_DIM, context);
    let prefix = context - rows as u32;
    let all = adapter
        .gather_kv_for_prefill_chunk_varlen(
            0,
            &bf16_array(&q, &[rows as i64, QUERY_HEADS_D256 as i64, D256_DIM as i64]),
            prefix,
            scale,
        )
        .unwrap_or_else(|error| panic!("context {context} rows {rows}: {error}"));
    let all = host(&all);
    assert!(
        all.iter().all(|value| value.is_finite()),
        "context {context}: non-finite output"
    );
    for &row in sampled {
        let one = adapter
            .gather_kv_for_prefill_chunk_varlen(
                0,
                &bf16_array(
                    &q[row * width..(row + 1) * width],
                    &[1, QUERY_HEADS_D256 as i64, D256_DIM as i64],
                ),
                prefix + row as u32,
                scale,
            )
            .unwrap();
        let one = host(&one);
        let got = &all[row * width..(row + 1) * width];
        assert!(
            got.iter()
                .zip(&one)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "context {context} row {row}: the {rows}-row read differs from the row alone"
        );
    }
}

/// An int8 prefill chunk reads past the 87,040-token context where one
/// launch's partials (2,048 rows x 24 heads x 512-token partitions x 256)
/// pass INT_MAX: the dispatch runs the rows in tiles of at most 2,048
/// row-partitions (Splash `PagedAttention::prefillWorkspace`), and every
/// row keeps the bits of its own one-row read; a 512-row chunk at 9,000
/// tokens (113-row tiles) too.
///
/// Mutation V1: the dispatch runs one tile → INT_MAX refusal → red.
#[test]
fn int8_varlen_prefill_reads_past_the_aux_cap() {
    let Some(mut adapter) = long_int8_adapter(90_000) else {
        return;
    };
    varlen_rows_match_single_rows(&mut adapter, 9_000, 512, &[0, 112, 113, 511]);
    varlen_rows_match_single_rows(&mut adapter, 90_000, 2048, &[0, 1000, 2047]);
}
