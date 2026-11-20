//! Decode reads (AR, one row and N lanes) of the keys-token-major page
//! against the same rows on the 16-token vLLM page: the block-16 bits, each
//! lane of an N-lane read as it is alone at every depth, the route taken by
//! the K plane's rank only, every other page on its own kernels.

use std::sync::Arc;
use std::time::Instant;

use napi::{Error, Result};

use super::{BF16_16, CHUNK, HEAD_DIM, HEADS, KV_HEADS, Page, VLLM16, VLLM32, bf16_rows, err};
use crate::array::{DType, MxArray};
use crate::transformer::Activations;
use crate::transformer::paged_int8_tests::adapter_with_layout;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::{mlx_paged_decode_test_probe_count, mlx_paged_decode_test_probe_reset};

const KTM: Page = Page::KeysTokenMajor;
const LANES: u32 = 4;
/// Rows of each long lane: past the deepest history (150,000).
const FILL: u32 = 74 * CHUNK as u32;
/// Rows of each short lane: the adapter's own reads below the striped rule.
const SHORT: u32 = 3_000;
/// The dense one-row striped rule (`grouped_qwen35_shape_matches`).
const STRIPED_FROM: u32 = 16_384;
/// Decode probe kinds (`paged_decode_probe_counter`).
const VLLM: usize = 0;
const STRIPED: usize = 1;
const GENERIC: usize = 2;
const V1: usize = 3;
const KINDS: usize = 4;
/// The keys-token-major one-row kernel from each history on
/// (`kKeysTokenMajorOneRow`; v1 up to one partition).
const ONE_ROW: [(u32, usize); 2] = [(1, V1), (513, STRIPED)];
/// Route hints of the decode launch.
const AUTO: u8 = 0;
const FORCE_GENERIC: u8 = 2;
/// Histories of the bit points: bf16 and INT8 queries (7,169 is the first
/// coded row), v1, generic, the one-row kernel borders, the striped rule,
/// stripes of even and odd block counts, histories ending in either half of
/// a page.
const HISTORIES: [u32; 14] = [
    400, 600, 2_049, 4_097, 7_169, 8_193, 12_289, 16_384, 16_401, 24_577, 40_013, 145_409, 145_425,
    150_000,
];
/// N rows within one partition: one v1 launch on both pages.
const V1_LANES: [u32; 4] = [400, 512, 17, 300];
/// Lane histories of the wave pins (2, 3 and 4 lanes): v1, the one-row
/// striped border, the one-row cells, both sides of the dense striped rule,
/// deep histories ending in either half of a page.
const LANE_HISTORIES: [u32; 15] = [
    400, 513, 600, 2_049, 8_193, 12_289, 16_383, 16_384, 16_385, 16_401, 24_577, 40_013, 145_409,
    145_425, 150_000,
];
/// Waves whose lanes stand at different depths: under one partition, between
/// it and the dense striped rule, past the rule; bf16 rows beside coded rows.
const MIXED_WAVES: [&[u32]; 7] = [
    &[400, 8_193, 16_385],
    &[512, 513],
    &[400, 16_385],
    &[7_169, 2_049, 600, 16_383],
    &[400, 12_289, 16_385, 513],
    &[145_409, 24_577, 2_049, 600],
    &[16_384, 150_000, 16_383, 513],
];

fn scale() -> f32 {
    1.0 / (HEAD_DIM as f32).sqrt()
}

fn one_row_kind(h: u32) -> usize {
    ONE_ROW
        .iter()
        .rev()
        .find(|&&(from, _)| h >= from)
        .map_or(V1, |&(_, kind)| kind)
}

/// The launches of an N-row read: one v1 launch when every row is within one
/// partition, else one launch per row, the row's own.
fn wave_kinds(lens: &[u32]) -> Vec<usize> {
    if lens.iter().all(|&h| h <= 512) {
        vec![V1]
    } else {
        lens.iter().map(|&h| one_row_kind(h)).collect()
    }
}

fn launches(kinds: &[usize]) -> [u64; KINDS] {
    let mut n = [0; KINDS];
    for &kind in kinds {
        n[kind] += 1;
    }
    n
}

/// The route hint under which the block-16 page reads one row at `h` in the
/// form the keys-token-major page takes there; `None` when no block-16 route
/// has that form.
fn block16_hint(h: u32) -> Option<u8> {
    let striped16 = h >= STRIPED_FROM;
    match one_row_kind(h) {
        STRIPED if !striped16 => None,
        GENERIC if striped16 => Some(FORCE_GENERIC),
        _ => Some(AUTO),
    }
}

/// Decode launches so far, by probe kind.
fn decode_counts() -> [u64; KINDS] {
    let mut counts = [0; KINDS];
    for (kind, count) in counts.iter_mut().enumerate() {
        // SAFETY: a plain counter read in the bridge.
        *count = unsafe { mlx_paged_decode_test_probe_count(kind as i32) };
    }
    counts
}

fn decode_delta(before: [u64; KINDS]) -> [u64; KINDS] {
    let after = decode_counts();
    std::array::from_fn(|i| after[i] - before[i])
}

struct Lanes {
    adapters: Vec<PagedKVCacheAdapter>,
    _pools: Vec<Arc<LayerKVPool>>,
}

/// One adapter per page, each holding `lanes` (sequence, rows) of one layer,
/// the same rows in every page, written through the product write path.
fn lanes_fixture(test: &str, pages: &[Page], lanes: &[(SeqId, u32)]) -> Result<Option<Lanes>> {
    if crate::test_support::paged_prefill_unsupported(test) {
        return Ok(None);
    }
    let mut adapters = Vec::new();
    let mut pools = Vec::new();
    for &page in pages {
        let block = page.block();
        let config = PagedAttentionConfig {
            block_size: block,
            gpu_memory_mb: 16_384,
            head_size: HEAD_DIM as u32,
            num_kv_heads: KV_HEADS as u32,
            num_layers: 1,
            max_seq_len: Some(262_144),
            max_batch_size: Some(lanes.len() as u32),
        };
        let blocks = lanes.iter().map(|&(_, rows)| rows / block + 2).sum::<u32>() + 64;
        let (mut adapter, pool) =
            adapter_with_layout(config, blocks, blocks, page.dtype(), page.layout())
                .ok_or_else(|| Error::from_reason(format!("no {page:?} pool")))?;
        adapter.set_activations(Activations::Auto);
        for &(seq, rows) in lanes {
            adapter.begin_request(seq).map_err(err)?;
            adapter.allocate_suffix_blocks(rows).map_err(err)?;
        }
        adapters.push(adapter);
        pools.push(pool);
    }
    for &(seq, rows) in lanes {
        let mut start = 0;
        while start < rows {
            let n = (rows - start).min(CHUNK as u32);
            let seed = 9_000 + 1_000 * u64::from(seq) + 2 * u64::from(start / CHUNK as u32);
            let k = bf16_rows(&[i64::from(n), KV_HEADS, HEAD_DIM], seed)?;
            let v = bf16_rows(&[i64::from(n), KV_HEADS, HEAD_DIM], seed + 1)?;
            let ids: Vec<u32> = (start..start + n)
                .map(|t| (seq * 7_919 + t) % 65_536)
                .collect();
            for adapter in adapters.iter_mut() {
                adapter.activate_request(seq).map_err(err)?;
                assert_eq!(adapter.current_token_count(), start);
                adapter.record_tokens(&ids).map_err(err)?;
                adapter
                    .update_keys_values_native(0, &k, &v, start)
                    .map_err(err)?;
                adapter.eval_pending_pool_writes().map_err(err)?;
            }
            start += n;
        }
    }
    Ok(Some(Lanes {
        adapters,
        _pools: pools,
    }))
}

fn long_lanes(rows: u32) -> Vec<(SeqId, u32)> {
    (1..=LANES).map(|seq| (seq, rows)).collect()
}

/// The planes and tables a decode read of `seqs` takes from its adapter.
struct Read {
    table: MxArray,
    planes: [MxArray; 4],
    block: u32,
    dtype: u8,
}

fn read_of(adapter: &mut PagedKVCacheAdapter, page: Page, seqs: &[SeqId]) -> Result<Read> {
    let (table, _, _) = adapter.decode_attention_inputs_batched(seqs).map_err(err)?;
    let planes = [
        adapter.key_pool_array(0).map_err(err)?,
        adapter.value_pool_array(0).map_err(err)?,
        adapter.k_scale_array(0).map_err(err)?,
        adapter.v_scale_array(0).map_err(err)?,
    ];
    Ok(Read {
        table,
        planes,
        block: page.block(),
        dtype: match page.dtype() {
            MetalDtype::Int8 => 3,
            _ => 1,
        },
    })
}

/// The queries of a decode read as the adapter makes them: rows at or past
/// `from` score INT8 codes and their score scales.
fn decode_inputs(
    from: Option<u32>,
    q: &MxArray,
    positions: &[u32],
) -> Result<(MxArray, Option<MxArray>)> {
    let Some(from) = from.filter(|&from| positions.iter().any(|&p| p >= from)) else {
        return Ok((q.clone(), None));
    };
    let positions: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let positions = MxArray::from_int32(&positions, &[positions.len() as i64])?;
    let mut codes = std::ptr::null_mut();
    let mut score_scales = std::ptr::null_mut();
    // SAFETY: both inputs are live arrays; the outputs are new on success.
    let ok = unsafe {
        mlx_sys::mlx_paged_query_codes_forward(
            q.as_raw_ptr(),
            positions.as_raw_ptr(),
            from,
            scale(),
            &mut codes,
            &mut score_scales,
        )
    };
    if !ok {
        return Err(Error::from_reason("query codes refused"));
    }
    Ok((
        MxArray::from_handle(codes, "ktm_decode_codes")?,
        Some(MxArray::from_handle(
            score_scales,
            "ktm_decode_score_scales",
        )?),
    ))
}

/// One decode launch as `gather_kv_for_decode_graph_batched_with_plan` makes
/// it, at histories `lens`, under route hint `hint`.
fn launch(
    read: &Read,
    q: &MxArray,
    score_scales: Option<&MxArray>,
    lens: &[u32],
    hint: u8,
) -> Result<MxArray> {
    let lens: Vec<i32> = lens.iter().map(|&h| h as i32).collect();
    let seq_lens = MxArray::from_int32(&lens, &[lens.len() as i64])?;
    // SAFETY: every pointer is a live MxArray handle; the output is new.
    let raw = unsafe {
        mlx_sys::mlx_paged_attention_forward_with_plan(
            q.as_raw_ptr(),
            read.planes[0].as_raw_ptr(),
            read.planes[1].as_raw_ptr(),
            read.table.as_raw_ptr(),
            seq_lens.as_raw_ptr(),
            read.planes[2].as_raw_ptr(),
            read.planes[3].as_raw_ptr(),
            scale(),
            0.0,
            0,
            read.block as i32,
            HEADS as i32,
            KV_HEADS as i32,
            HEAD_DIM as i32,
            read.dtype,
            hint,
            0,
            score_scales.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
        )
    };
    if raw.is_null() {
        return Err(Error::from_reason("decode launch refused"));
    }
    MxArray::from_handle(raw, "ktm_decode")
}

/// The read of `read` at histories `lens` (one per row) under route hint
/// `hint`, its output bits and launches.
fn attend(
    read: &Read,
    from: Option<u32>,
    q: &MxArray,
    lens: &[u32],
    hint: u8,
) -> Result<(Vec<u16>, [u64; KINDS])> {
    let positions: Vec<u32> = lens.iter().map(|&h| h - 1).collect();
    let (q, score_scales) = decode_inputs(from, q, &positions)?;
    let before = decode_counts();
    let out = launch(read, &q, score_scales.as_ref(), lens, hint)?;
    out.eval();
    Ok((out.to_uint16_native()?, decode_delta(before)))
}

fn row_queries(rows: usize) -> Result<MxArray> {
    bf16_rows(&[rows as i64, HEADS, HEAD_DIM], 60 + rows as u64)
}

/// The query row `row` of `rows` queries, as a one-row query.
fn one_row(q: &MxArray, row: usize) -> Result<MxArray> {
    let x = q
        .slice_axis(0, row as i64, row as i64 + 1)?
        .reshape(&[1, HEADS, HEAD_DIM])?;
    x.eval();
    Ok(x)
}

/// (elements whose bits differ, NaNs in `b`, exact zeros in `b`).
fn differ(a: &[u16], b: &[u16]) -> (usize, usize, usize) {
    assert_eq!(a.len(), b.len(), "compared outputs differ in size");
    let mut n = (0, 0, 0);
    for (&x, &y) in a.iter().zip(b) {
        n.0 += usize::from(x != y);
        n.1 += usize::from(y & 0x7f80 == 0x7f80 && y & 0x007f != 0);
        n.2 += usize::from(y & 0x7fff == 0);
    }
    n
}

fn row_bits(bits: &[u16], row: usize) -> &[u16] {
    let n = (HEADS * HEAD_DIM) as usize;
    &bits[row * n..(row + 1) * n]
}

fn check(failures: &mut Vec<String>, what: String, a: &[u16], b: &[u16]) {
    let (d, nans, zeros) = differ(a, b);
    println!(
        "ROW ktm_decode_bits {what} differ={d} nans={nans} zeros={zeros} elements={}",
        b.len()
    );
    if d != 0 || nans != 0 || zeros * 100 >= b.len() {
        failures.push(format!("{what}: differ {d} nans {nans} zeros {zeros}"));
    }
}

/// The one-row decode at every history, and the N-row decode within one
/// partition (one v1 launch on both pages), give the bits of the block-16
/// route of the same form; the direct launch is the adapter's.
#[test]
fn keys_token_major_decode_is_the_block16_decode() -> Result<()> {
    let mut lanes = long_lanes(FILL);
    lanes.extend((LANES + 1..=2 * LANES).map(|seq| (seq, SHORT)));
    let pages = [VLLM16, KTM];
    let Some(mut f) = lanes_fixture(
        "keys_token_major_decode_is_the_block16_decode",
        &pages,
        &lanes,
    )?
    else {
        return Ok(());
    };
    let from = f.adapters[0].int8_queries_from();
    let long: Vec<SeqId> = (1..=LANES).collect();
    let short: Vec<SeqId> = (LANES + 1..=2 * LANES).collect();
    let mut failures = Vec::new();

    for (p, &page) in pages.iter().enumerate() {
        let adapter = &mut f.adapters[p];
        for (seqs, rows) in [
            (&long[..1], FILL),
            (&long[..], FILL),
            (&short[..1], SHORT),
            (&short[..], SHORT),
        ] {
            let q = row_queries(seqs.len())?;
            let by_adapter = if seqs.len() == 1 {
                adapter.activate_request(seqs[0]).map_err(err)?;
                adapter.gather_kv_for_decode_graph(0, &q, scale(), 1.0)
            } else {
                adapter.gather_kv_for_decode_graph_batched(0, &q, seqs, scale(), 1.0)
            }
            .map_err(err)?;
            by_adapter.eval();
            let read = read_of(adapter, page, seqs)?;
            let (direct, _) = attend(&read, from, &q, &vec![rows; seqs.len()], AUTO)?;
            check(
                &mut failures,
                format!(
                    "page={page:?} rows={} h={rows} direct-vs-adapter",
                    seqs.len()
                ),
                &by_adapter.to_uint16_native()?,
                &direct,
            );
        }
    }

    let reads: Vec<[Read; 2]> = pages
        .iter()
        .zip(f.adapters.iter_mut())
        .map(|(&page, adapter)| {
            Ok([
                read_of(adapter, page, &long[..1])?,
                read_of(adapter, page, &long)?,
            ])
        })
        .collect::<Result<_>>()?;
    let mut points: Vec<Vec<u32>> = HISTORIES.iter().map(|&h| vec![h]).collect();
    points.push(vec![400; LANES as usize]);
    points.push(V1_LANES.to_vec());
    for lens in points {
        let rows = lens.len();
        let r = usize::from(rows > 1);
        let hint = if rows == 1 {
            block16_hint(lens[0])
        } else {
            Some(AUTO)
        };
        let Some(hint) = hint else {
            println!(
                "ROW ktm_decode_bits rows={rows} lens={lens:?} no block-16 route of this form (INFO)"
            );
            continue;
        };
        let q = row_queries(rows)?;
        let (c16, _) = attend(&reads[0][r], from, &q, &lens, hint)?;
        let (b32, _) = attend(&reads[1][r], from, &q, &lens, AUTO)?;
        check(
            &mut failures,
            format!("rows={rows} lens={lens:?} b32-vs-c16"),
            &c16,
            &b32,
        );
    }
    assert!(
        failures.is_empty(),
        "keys-token-major decode bits: {failures:?}"
    );
    Ok(())
}

/// Every lane of an N-row decode (2, 3 and 4 lanes, at one depth or at
/// mixed depths) reads the bits it reads alone, which are the block-16 row
/// alone where the block-16 page has that form; against the block-16 N-row
/// read the bits are INFO.
#[test]
fn keys_token_major_lanes_read_alone() -> Result<()> {
    let pages = [VLLM16, KTM];
    let Some(mut f) = lanes_fixture(
        "keys_token_major_lanes_read_alone",
        &pages,
        &long_lanes(FILL),
    )?
    else {
        return Ok(());
    };
    let from = f.adapters[0].int8_queries_from();
    let long: Vec<SeqId> = (1..=LANES).collect();
    let mut waves = Vec::new();
    let mut alone = Vec::new();
    for (&page, adapter) in pages.iter().zip(f.adapters.iter_mut()) {
        waves.push(
            (2..=long.len())
                .map(|n| read_of(adapter, page, &long[..n]))
                .collect::<Result<Vec<_>>>()?,
        );
        alone.push(
            long.iter()
                .map(|&seq| read_of(adapter, page, &[seq]))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    let mut points: Vec<Vec<u32>> = Vec::new();
    for n in 2..=long.len() {
        points.extend(LANE_HISTORIES.iter().map(|&h| vec![h; n]));
    }
    points.extend(MIXED_WAVES.iter().map(|lens| lens.to_vec()));
    let mut failures = Vec::new();
    for lens in points {
        let n = lens.len();
        let q = row_queries(n)?;
        let (b32, _) = attend(&waves[1][n - 2], from, &q, &lens, AUTO)?;
        let (c16, _) = attend(&waves[0][n - 2], from, &q, &lens, AUTO)?;
        let (info, _, _) = differ(&c16, &b32);
        println!("ROW ktm_decode_lanes lens={lens:?} b32-vs-c16-lanes differ={info} (INFO)");
        for (row, &h) in lens.iter().enumerate() {
            let q1 = one_row(&q, row)?;
            let (alone_b32, _) = attend(&alone[1][row], from, &q1, &[h], AUTO)?;
            check(
                &mut failures,
                format!("lens={lens:?} row={row} b32-lane-vs-b32-alone"),
                &alone_b32,
                row_bits(&b32, row),
            );
            let Some(hint) = block16_hint(h) else {
                println!(
                    "ROW ktm_decode_lanes lens={lens:?} row={row} no block-16 route of this form (INFO)"
                );
                continue;
            };
            let (alone_c16, _) = attend(&alone[0][row], from, &q1, &[h], hint)?;
            check(
                &mut failures,
                format!("lens={lens:?} row={row} b32-lane-vs-c16-alone"),
                &alone_c16,
                row_bits(&b32, row),
            );
        }
    }
    assert!(failures.is_empty(), "lanes read alone: {failures:?}");
    Ok(())
}

/// A direct decode launch over fabricated planes: whether the validator takes
/// them. `k_shape` / `v_shape` of `dtype`, `heads` query heads over 4 KV
/// heads, `head` dimensions, `block` tokens per block, one row at history 1.
fn decode_takes(
    dtype: DType,
    k_shape: &[i64],
    v_shape: &[i64],
    heads: i64,
    head: i64,
    block: i64,
) -> Result<bool> {
    let blocks = k_shape[0];
    let int8 = dtype == DType::Int8;
    let scale_shape: Vec<i64> = if int8 {
        vec![blocks, KV_HEADS, block]
    } else {
        vec![1]
    };
    let read = Read {
        table: MxArray::from_int32(&[0], &[1, 1])?,
        planes: [
            MxArray::zeros(k_shape, Some(dtype))?,
            MxArray::zeros(v_shape, Some(dtype))?,
            MxArray::zeros(&scale_shape, Some(DType::Float32))?,
            MxArray::zeros(&scale_shape, Some(DType::Float32))?,
        ],
        block: block as u32,
        dtype: if int8 { 3 } else { 1 },
    };
    let q = MxArray::zeros(&[1, heads, head], Some(DType::BFloat16))?;
    let lens = MxArray::from_int32(&[1], &[1])?;
    // SAFETY: every pointer is a live MxArray handle; the output is new.
    let raw = unsafe {
        mlx_sys::mlx_paged_attention_forward_with_plan(
            q.as_raw_ptr(),
            read.planes[0].as_raw_ptr(),
            read.planes[1].as_raw_ptr(),
            read.table.as_raw_ptr(),
            lens.as_raw_ptr(),
            read.planes[2].as_raw_ptr(),
            read.planes[3].as_raw_ptr(),
            1.0,
            0.0,
            0,
            block as i32,
            heads as i32,
            KV_HEADS as i32,
            head as i32,
            read.dtype,
            0,
            0,
            std::ptr::null_mut(),
        )
    };
    if raw.is_null() {
        return Ok(false);
    }
    MxArray::from_handle(raw, "ktm_decode_refusal")?;
    Ok(true)
}

/// Whether the ragged (varlen) read takes a keys-token-major page.
fn varlen_takes_the_page() -> Result<bool> {
    let nb = 4;
    let planes = [
        MxArray::zeros(&[nb, KV_HEADS, 32, HEAD_DIM], Some(DType::Int8))?,
        MxArray::zeros(&[nb, KV_HEADS, HEAD_DIM, 32], Some(DType::Int8))?,
        MxArray::zeros(&[nb, KV_HEADS, 32], Some(DType::Float32))?,
        MxArray::zeros(&[nb, KV_HEADS, 32], Some(DType::Float32))?,
    ];
    let q = MxArray::zeros(&[2, HEADS, HEAD_DIM], Some(DType::BFloat16))?;
    let table = MxArray::from_int32(&[0], &[1, 1])?;
    let lens = MxArray::from_int32(&[2], &[1])?;
    let cu = MxArray::from_int32(&[0, 2], &[2])?;
    // SAFETY: every pointer is a live MxArray handle; the output is new.
    let raw = unsafe {
        mlx_sys::mlx_paged_attention_varlen_forward(
            q.as_raw_ptr(),
            planes[0].as_raw_ptr(),
            planes[1].as_raw_ptr(),
            table.as_raw_ptr(),
            lens.as_raw_ptr(),
            cu.as_raw_ptr(),
            planes[2].as_raw_ptr(),
            planes[3].as_raw_ptr(),
            1.0,
            0.0,
            0,
            32,
            HEADS as i32,
            KV_HEADS as i32,
            HEAD_DIM as i32,
            3,
            std::ptr::null_mut(),
        )
    };
    if raw.is_null() {
        return Ok(false);
    }
    MxArray::from_handle(raw, "ktm_varlen_refusal")?;
    Ok(true)
}

/// On the keys-token-major page every decode read takes a new kernel
/// (v1, generic, striped; one v1 launch when every row is within one
/// partition, else one launch per row, the row's own) and none a vLLM-page kernel; the
/// vLLM pages at block 16 and 32 keep theirs; the validator takes a rank-4 K
/// only as the whole page; the ragged read refuses the page.
#[test]
fn decode_route_follows_the_key_rank() -> Result<()> {
    let fill = 13 * CHUNK as u32;
    let pages = [VLLM16, VLLM32, KTM];
    let Some(mut f) = lanes_fixture(
        "decode_route_follows_the_key_rank",
        &pages,
        &long_lanes(fill),
    )?
    else {
        return Ok(());
    };
    // SAFETY: a plain counter reset in the bridge.
    unsafe { mlx_paged_decode_test_probe_reset() };
    let from = f.adapters[0].int8_queries_from();
    let long: Vec<SeqId> = (1..=LANES).collect();
    let one_row_points = [
        400, 512, 513, 2_049, 4_096, 4_097, 8_192, 8_193, 12_288, 12_289, 16_383, 16_384, 24_576,
        24_577,
    ]
    .map(|h| (h, launches(&[one_row_kind(h)])));
    let lane_points: Vec<([u32; 4], [u64; KINDS])> = [
        [400; 4],
        [512; 4],
        [513; 4],
        [2_049; 4],
        [16_383; 4],
        [24_577; 4],
        [24_577, 2_049, 400, 16_384],
        [512, 513, 400, 600],
    ]
    .into_iter()
    .map(|lens| (lens, launches(&wave_kinds(&lens))))
    .collect();
    let mut one_row_failures = Vec::new();
    let mut lane_failures = Vec::new();
    for (&page, adapter) in pages.iter().zip(f.adapters.iter_mut()) {
        let vllm = launches(&[VLLM]);
        let one = read_of(adapter, page, &long[..1])?;
        for (h, want) in one_row_points {
            let want = if page == KTM { want } else { vllm };
            let (_, got) = attend(&one, from, &row_queries(1)?, &[h], AUTO)?;
            println!(
                "ROW ktm_decode_route page={page:?} lens=[{h}] launches={got:?} want={want:?}"
            );
            if got != want {
                one_row_failures.push(format!("{page:?} [{h}]: {got:?}, want {want:?}"));
            }
        }
        let all = read_of(adapter, page, &long)?;
        for &(lens, want) in &lane_points {
            let want = if page == KTM { want } else { vllm };
            let (_, got) = attend(&all, from, &row_queries(lens.len())?, &lens, AUTO)?;
            println!(
                "ROW ktm_decode_route page={page:?} lens={lens:?} launches={got:?} want={want:?}"
            );
            if got != want {
                lane_failures.push(format!("{page:?} {lens:?}: {got:?}, want {want:?}"));
            }
        }
    }
    assert!(
        one_row_failures.is_empty(),
        "one-row decode route: {one_row_failures:?}"
    );
    assert!(
        lane_failures.is_empty(),
        "N-row decode route: {lane_failures:?}"
    );

    let nb = 4;
    let (h, d) = (HEADS, HEAD_DIM);
    let whole = decode_takes(
        DType::Int8,
        &[nb, KV_HEADS, 32, d],
        &[nb, KV_HEADS, d, 32],
        h,
        d,
        32,
    )?;
    let refusals = [
        (
            "token-major V",
            decode_takes(
                DType::Int8,
                &[nb, KV_HEADS, 32, d],
                &[nb, KV_HEADS, 32, d],
                h,
                d,
                32,
            )?,
        ),
        (
            "block 16",
            decode_takes(
                DType::Int8,
                &[nb, KV_HEADS, 16, d],
                &[nb, KV_HEADS, d, 16],
                h,
                d,
                16,
            )?,
        ),
        (
            "bf16 page",
            decode_takes(
                DType::BFloat16,
                &[nb, KV_HEADS, 32, d],
                &[nb, KV_HEADS, d, 32],
                h,
                d,
                32,
            )?,
        ),
        (
            "group 8",
            decode_takes(
                DType::Int8,
                &[nb, KV_HEADS, 32, d],
                &[nb, KV_HEADS, d, 32],
                32,
                d,
                32,
            )?,
        ),
        (
            "D 128",
            decode_takes(
                DType::Int8,
                &[nb, KV_HEADS, 32, 128],
                &[nb, KV_HEADS, 128, 32],
                h,
                128,
                32,
            )?,
        ),
        (
            "rank-4 K beside a rank-5 V",
            decode_takes(
                DType::Int8,
                &[nb, KV_HEADS, 32, d],
                &[nb, KV_HEADS, d / 16, 32, 16],
                h,
                d,
                32,
            )?,
        ),
    ];
    let varlen = varlen_takes_the_page()?;
    println!("ROW ktm_decode_validator whole={whole} refusals={refusals:?} varlen_takes={varlen}");
    let taken: Vec<&str> = refusals
        .iter()
        .filter(|(_, ok)| *ok)
        .map(|(name, _)| *name)
        .collect();
    assert!(
        whole && taken.is_empty() && !varlen,
        "decode validator: whole page {whole}, took {taken:?}, varlen took the page {varlen}"
    );
    Ok(())
}

/// Every page other families build (INT8 vLLM at block 16 and 32,
/// bf16 at block 16) keeps the vLLM-page decode dispatch: one vLLM-page call per
/// read, no keys-token-major launch.
#[test]
fn other_pages_keep_the_product_decode() -> Result<()> {
    let fill = 10 * CHUNK as u32;
    let pages = [VLLM16, VLLM32, BF16_16];
    let Some(mut f) = lanes_fixture(
        "other_pages_keep_the_product_decode",
        &pages,
        &long_lanes(fill),
    )?
    else {
        return Ok(());
    };
    let long: Vec<SeqId> = (1..=LANES).collect();
    let mut failures = Vec::new();
    for (&page, adapter) in pages.iter().zip(f.adapters.iter_mut()) {
        let from = adapter.int8_queries_from();
        let reads = [
            read_of(adapter, page, &long[..1])?,
            read_of(adapter, page, &long)?,
        ];
        let points: [&[u32]; 6] = [
            &[600],
            &[2_049],
            &[16_384],
            &[20_000],
            &[2_049; 4],
            &[20_000; 4],
        ];
        for lens in points {
            let read = &reads[usize::from(lens.len() > 1)];
            let (bits, got) = attend(read, from, &row_queries(lens.len())?, lens, AUTO)?;
            let (_, nans, zeros) = differ(&bits, &bits);
            println!(
                "ROW ktm_decode_other page={page:?} lens={lens:?} launches={got:?} nans={nans} zeros={zeros}"
            );
            if got != launches(&[VLLM]) || nans != 0 || zeros * 100 >= bits.len() {
                failures.push(format!(
                    "{page:?} {lens:?}: launches {got:?}, nans {nans}, zeros {zeros}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "other pages decode: {failures:?}");
    Ok(())
}

/// Seconds on CLOCK_UPTIME_RAW, the clock of the command trace's `submitCpu`.
fn uptime() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid out-pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// One write per marker: the trace's completion thread writes stderr too.
fn mark(words: &str) {
    use std::io::Write;
    let line = format!("M0 {words} t={:.9}\n", uptime());
    let _ = std::io::stderr().lock().write_all(line.as_bytes());
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A decode timing row (run by hand with `MLX_METAL_COMMAND_TRACE=1`; the
/// GPU busy time per call comes from the command trace between the `M0 s`
/// markers): `lane_counts` rows at `depths`, the keys-token-major page
/// against the block-16 page, 40 rounds of both arms, start arm rotated, each
/// sample a chain of dependent calls. Bits before time: one row = the
/// block-16 row where that form exists; each lane of N rows = the lane alone.
fn timing_row(test: &str, lane_counts: &[usize], depths: &[u32]) -> Result<()> {
    const ROUNDS: usize = 40;
    const SAMPLE_MS: f64 = 6.0;
    const HEAT_S: f64 = 8.0;
    let pages = [VLLM16, KTM];
    let Some(mut f) = lanes_fixture(test, &pages, &long_lanes(FILL))? else {
        return Ok(());
    };
    let from = f.adapters[0].int8_queries_from();
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    println!(
        "ROW ktm_time setup int8 queries from {from:?} clocks uptime={:.3} epoch={epoch:.3}",
        uptime()
    );
    let long: Vec<SeqId> = (1..=LANES).collect();
    let mut reads = Vec::new();
    for (&page, adapter) in pages.iter().zip(f.adapters.iter_mut()) {
        let mut by_rows = Vec::new();
        for rows in 1..=LANES as usize {
            by_rows.push(read_of(adapter, page, &long[..rows])?);
        }
        reads.push(by_rows);
    }
    let alone = long
        .iter()
        .map(|&seq| read_of(&mut f.adapters[1], KTM, &[seq]))
        .collect::<Result<Vec<_>>>()?;
    let chain = |arm: usize, rows: usize, h: u32, q0: &MxArray, reps: usize| -> Result<MxArray> {
        let positions = vec![h - 1; rows];
        let lens = vec![h; rows];
        let mut x = q0.clone();
        for _ in 0..reps {
            let (q, score_scales) = decode_inputs(from, &x, &positions)?;
            x = launch(
                &reads[arm][rows - 1],
                &q,
                score_scales.as_ref(),
                &lens,
                AUTO,
            )?;
        }
        Ok(x)
    };

    for &rows in lane_counts {
        for &h in depths {
            let q = row_queries(rows)?;
            let lens = vec![h; rows];
            let hint = if rows == 1 {
                block16_hint(h)
            } else {
                Some(AUTO)
            };
            let (c16, _) = attend(&reads[0][rows - 1], from, &q, &lens, hint.unwrap_or(AUTO))?;
            let (b32, _) = attend(&reads[1][rows - 1], from, &q, &lens, AUTO)?;
            let (d, nans, _) = differ(&c16, &b32);
            println!("ROW ktm_time_bits rows={rows} h={h} differ={d} nans={nans}");
            assert!(nans == 0, "timing row: NaN at rows {rows} h {h}");
            if rows == 1 && hint.is_some() {
                assert_eq!(d, 0, "timing row bits: rows {rows} h {h}");
            }
            for (row, read) in alone.iter().enumerate().take(rows) {
                let (lane, _) = attend(read, from, &one_row(&q, row)?, &[h], AUTO)?;
                let (d, _, _) = differ(&lane, row_bits(&b32, row));
                println!("ROW ktm_time_lane_bits rows={rows} h={h} row={row} differ={d}");
                assert_eq!(d, 0, "timing row lane bits: rows {rows} h {h} row {row}");
            }
        }
    }

    let heat = Instant::now();
    let q4 = row_queries(LANES as usize)?;
    while heat.elapsed().as_secs_f64() < HEAT_S {
        let x = chain(0, LANES as usize, 24_577, &q4, 8)?;
        x.eval();
    }
    for &rows in lane_counts {
        for &h in depths {
            let q0 = row_queries(rows)?;
            let mut probe_ms = f64::MAX;
            for _ in 0..3 {
                let x = chain(0, rows, h, &q0, 4)?;
                let t = Instant::now();
                x.eval();
                probe_ms = probe_ms.min(t.elapsed().as_secs_f64() * 1e3 / 4.0);
            }
            let reps = ((SAMPLE_MS / probe_ms) as usize).clamp(4, 256);
            for arm in 0..2 {
                for _ in 0..2 {
                    chain(arm, rows, h, &q0, reps)?.eval();
                }
            }
            let mut walls = [Vec::new(), Vec::new()];
            for round in 0..ROUNDS {
                for j in 0..2 {
                    let arm = (round + j) % 2;
                    let x = chain(arm, rows, h, &q0, reps)?;
                    let tag = format!("n{rows}.h{h}.{}.{round}", ["C16", "B32"][arm]);
                    mark(&format!("s begin {tag}"));
                    let t = Instant::now();
                    x.eval();
                    let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
                    mark(&format!("s end {tag}"));
                    walls[arm].push(ms);
                }
            }
            let paired: Vec<f64> = walls[1][1..]
                .iter()
                .zip(&walls[0][1..])
                .map(|(b, c)| b / c)
                .collect();
            println!(
                "ROW ktm_time rows={rows} h={h} reps={reps} wall_c16_ms={:.4} wall_b32_ms={:.4} wall_paired={:.4} (INFO)",
                median(walls[0][1..].to_vec()),
                median(walls[1][1..].to_vec()),
                median(paired)
            );
        }
    }
    Ok(())
}

/// 1 to 4 rows at 2,049 / 8,193 / 24,577 / 145,409.
#[test]
#[ignore = "timing row: run by hand under the command trace"]
fn keys_token_major_decode_timing() -> Result<()> {
    timing_row(
        "keys_token_major_decode_timing",
        &[1, 2, 3, 4],
        &[2_049, 8_193, 24_577, 145_409],
    )
}

/// 2 to 4 lanes at 2,049 / 4,097 / 8,193 / 12,289: one launch per lane.
#[test]
#[ignore = "timing row: run by hand under the command trace"]
fn keys_token_major_wave_timing() -> Result<()> {
    timing_row(
        "keys_token_major_wave_timing",
        &[2, 3, 4],
        &[2_049, 4_097, 8_193, 12_289],
    )
}
