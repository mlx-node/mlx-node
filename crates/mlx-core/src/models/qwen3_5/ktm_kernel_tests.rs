//! The in-place prefill / verify tile over the keys-token-major page (INT8,
//! 32-token blocks, heads 24/4, D 256) against the product tile over the
//! 16-token vLLM page that holds the same rows: same bits, the route taken by
//! the K plane's rank only, and every other page on its own entry.

use std::sync::Arc;

use napi::{Error, Result};

use crate::array::{DType, MxArray};
use crate::transformer::Activations;
use crate::transformer::paged_int8_tests::adapter_with_layout;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::prefill_plan::{self, Pass};
use mlx_paged_attn::{LayerKVPool, PageLayout, PagedAttentionConfig};

mod decode;
mod family_routes;

const HEADS: i64 = 24;
const KV_HEADS: i64 = 4;
const HEAD_DIM: i64 = 256;
const CHUNK: i64 = 2048;
const VERIFY_ROWS: i64 = 8;
/// Rows stored in every pool: past the deepest point (145,592 visible), so a
/// tail page's rows past the visible tokens hold stored rows.
const POOL_TOKENS: u32 = 72 * CHUNK as u32;
/// The product's INT8-query position on Apple10 (`prefill_plan::int8_query_from`).
const PRODUCT_INT8_FROM: u32 = 256;

/// (pass, committed, rows) of the bit points.
const BIT_POINTS: [(Pass, u32, i64); 8] = [
    (Pass::Prefill, 0, CHUNK),
    (Pass::Prefill, 1_000, CHUNK),
    (Pass::Prefill, 6_144, CHUNK),
    (Pass::Prefill, 65_536, CHUNK),
    (Pass::Prefill, 131_072, CHUNK),
    (Pass::Verify, 25_000, VERIFY_ROWS),
    (Pass::Verify, 40_013, VERIFY_ROWS),
    (Pass::Verify, 145_584, VERIFY_ROWS),
];

/// Points whose last page holds 16 visible rows (2,032 and 145,584 visible).
const TAIL_POINTS: [(Pass, u32, i64); 2] = [
    (Pass::Prefill, 0, 2_032),
    (Pass::Verify, 145_576, VERIFY_ROWS),
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Page {
    /// The product page: vLLM K x-pack, `block` tokens per block.
    Vllm { block: u32, dtype: MetalDtype },
    /// Keys token-major, values dimension-major, INT8, 32-token blocks.
    KeysTokenMajor,
}

const VLLM16: Page = Page::Vllm {
    block: 16,
    dtype: MetalDtype::Int8,
};
const VLLM32: Page = Page::Vllm {
    block: 32,
    dtype: MetalDtype::Int8,
};
const BF16_16: Page = Page::Vllm {
    block: 16,
    dtype: MetalDtype::BFloat16,
};

impl Page {
    fn block(self) -> u32 {
        match self {
            Page::Vllm { block, .. } => block,
            Page::KeysTokenMajor => 32,
        }
    }

    fn dtype(self) -> MetalDtype {
        match self {
            Page::Vllm { dtype, .. } => dtype,
            Page::KeysTokenMajor => MetalDtype::Int8,
        }
    }

    fn layout(self) -> PageLayout {
        PageLayout {
            keys_token_major: self == Page::KeysTokenMajor,
            ..PageLayout::default()
        }
    }
}

struct Fixture {
    adapters: Vec<PagedKVCacheAdapter>,
    _pools: Vec<Arc<LayerKVPool>>,
}

fn err(e: String) -> Error {
    Error::from_reason(e)
}

fn bf16_rows(shape: &[i64], seed: u64) -> Result<MxArray> {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let x =
        MxArray::random_normal(shape, 0.0, 1.0, Some(DType::Float32))?.astype(DType::BFloat16)?;
    x.eval();
    Ok(x)
}

/// One adapter per page, each holding the same `POOL_TOKENS` rows of one
/// layer, written through the product write path. `None` where the device
/// has no tile.
fn fixture(test: &str, pages: &[Page]) -> Result<Option<Fixture>> {
    if crate::test_support::paged_prefill_unsupported(test) {
        return Ok(None);
    }
    assert!(
        crate::transformer::paged_kv_cache_adapter::device_prefill_mpp_builds(),
        "this GPU attends through the portable kernel"
    );
    let mut adapters = Vec::new();
    let mut pools = Vec::new();
    for &page in pages {
        let block = page.block();
        let config = PagedAttentionConfig {
            block_size: block,
            gpu_memory_mb: 8192,
            head_size: HEAD_DIM as u32,
            num_kv_heads: KV_HEADS as u32,
            num_layers: 1,
            max_seq_len: Some(262_144),
            max_batch_size: Some(1),
        };
        let blocks = POOL_TOKENS / block + 64;
        let (mut adapter, pool) =
            adapter_with_layout(config, blocks, blocks, page.dtype(), page.layout())
                .ok_or_else(|| Error::from_reason(format!("no {page:?} pool")))?;
        adapter.reset_for_new_request(1).map_err(err)?;
        adapter.allocate_suffix_blocks(POOL_TOKENS).map_err(err)?;
        adapters.push(adapter);
        pools.push(pool);
    }
    for chunk in 0..POOL_TOKENS / CHUNK as u32 {
        let seed = 7_000 + 2 * u64::from(chunk);
        let k = bf16_rows(&[CHUNK, KV_HEADS, HEAD_DIM], seed)?;
        let v = bf16_rows(&[CHUNK, KV_HEADS, HEAD_DIM], seed + 1)?;
        let start = chunk * CHUNK as u32;
        let ids: Vec<u32> = (start..start + CHUNK as u32).collect();
        for adapter in adapters.iter_mut() {
            assert_eq!(adapter.current_token_count(), start);
            adapter.record_tokens(&ids).map_err(err)?;
            adapter
                .update_keys_values_native(0, &k, &v, start)
                .map_err(err)?;
            adapter.eval_pending_pool_writes().map_err(err)?;
        }
    }
    Ok(Some(Fixture {
        adapters,
        _pools: pools,
    }))
}

/// `[prefill, verify, q8q, keys-token-major]` tile launches so far.
fn tile_counts() -> [u64; 4] {
    // SAFETY: plain counter reads in the bridge.
    unsafe {
        [
            mlx_sys::mlx_paged_prefill_test_probe_count(0),
            mlx_sys::mlx_paged_prefill_test_probe_count(1),
            mlx_sys::mlx_paged_prefill_test_probe_count(3),
            mlx_sys::mlx_paged_prefill_test_probe_count(4),
        ]
    }
}

fn delta(before: [u64; 4], after: [u64; 4]) -> [u64; 4] {
    [0, 1, 2, 3].map(|i| after[i] - before[i])
}

/// Queries of a point: the prefill queries' first `rows` rows, or the
/// verify rows.
fn queries(pass: Pass, rows: i64) -> Result<MxArray> {
    let q = match pass {
        Pass::Prefill => bf16_rows(&[CHUNK, HEADS, HEAD_DIM], 52)?,
        Pass::Verify => bf16_rows(&[VERIFY_ROWS, HEADS, HEAD_DIM], 54)?,
    };
    if q.shape_at(0)? == rows {
        return Ok(q);
    }
    let q = q.slice_axis(0, 0, rows)?;
    q.eval();
    Ok(q)
}

/// One product attention of the point and the tile launches it made:
/// `(output, [prefill, verify, q8q, keys-token-major] launches, [bf16, int8]
/// query launches of the adapter)`.
fn attend(
    adapter: &mut PagedKVCacheAdapter,
    q: &MxArray,
    pass: Pass,
    committed: u32,
) -> Result<(MxArray, [u64; 4], [u64; 2])> {
    let queries_before = adapter.tile_query_launches();
    let before = tile_counts();
    let out = adapter
        .paged_prefill_attention(0, q, committed, pass, 1.0 / (HEAD_DIM as f32).sqrt())
        .map_err(err)?;
    out.eval();
    let after = adapter.tile_query_launches();
    Ok((
        out,
        delta(before, tile_counts()),
        [after[0] - queries_before[0], after[1] - queries_before[1]],
    ))
}

/// (elements whose bits differ, NaNs in `b`, exact zeros in `b`, elements).
fn compare(a: &MxArray, b: &MxArray) -> Result<(usize, usize, usize, usize)> {
    let (a, b) = (a.to_uint16_native()?, b.to_uint16_native()?);
    assert_eq!(a.len(), b.len(), "compared outputs differ in size");
    let mut differ = 0;
    let mut nans = 0;
    let mut zeros = 0;
    for (&x, &y) in a.iter().zip(&b) {
        if y & 0x7f80 == 0x7f80 && y & 0x007f != 0 {
            nans += 1;
        }
        if y & 0x7fff == 0 {
            zeros += 1;
        }
        if x != y {
            differ += 1;
        }
    }
    Ok((differ, nans, zeros, b.len()))
}

/// The query mode of a run: bf16 queries (`q8`), INT8 queries from row 0
/// (`q8q`), or the product's position rule.
#[derive(Clone, Copy, Debug)]
enum Queries {
    Bf16,
    Int8,
    Product,
}

fn set_queries(adapter: &mut PagedKVCacheAdapter, mode: Queries) {
    match mode {
        Queries::Bf16 => adapter.set_activations(Activations::Bf16),
        Queries::Int8 => {
            adapter.set_activations(Activations::Auto);
            adapter.set_int8_query_threshold(0);
        }
        Queries::Product => {
            adapter.set_activations(super::load_plan::default_activations(false, true))
        }
    }
}

fn apple10() -> bool {
    // SAFETY: a plain read of a device fact.
    prefill_plan::int8_query_from(unsafe { mlx_sys::mlx_gpu_architecture_gen() })
        == Some(PRODUCT_INT8_FROM)
}

/// Every point of `points` in `mode` through the product adapter of both
/// pages: the keys-token-major output against the block-16 output, and the
/// launches each made. Appends a line per defect to `failures`.
fn compare_points(
    fixture: &mut Fixture,
    mode: Queries,
    points: &[(Pass, u32, i64)],
    failures: &mut Vec<String>,
) -> Result<()> {
    for adapter in fixture.adapters.iter_mut() {
        set_queries(adapter, mode);
    }
    for &(pass, committed, rows) in points {
        let q = queries(pass, rows)?;
        let (reference, ref_launches, ref_queries) =
            attend(&mut fixture.adapters[0], &q, pass, committed)?;
        let (tm, tm_launches, tm_queries) = attend(&mut fixture.adapters[1], &q, pass, committed)?;
        let (differ, nans, zeros, elements) = compare(&reference, &tm)?;
        let (_, ref_nans, ref_zeros, _) = compare(&reference, &reference)?;
        println!(
            "ROW ktm_bits q={mode:?} pass={pass:?} committed={committed} rows={rows} differ={differ} nans={nans} zeros={zeros} reference_zeros={ref_zeros} elements={elements} launches={tm_launches:?} queries={tm_queries:?}"
        );
        if differ != 0 || nans != 0 || ref_nans != 0 || ref_zeros * 100 >= elements {
            failures.push(format!(
                "{mode:?} {pass:?} {committed}+{rows}: differ {differ} nans {nans}/{ref_nans} zeros {ref_zeros}"
            ));
        }
        let launches = tm_queries[0] + tm_queries[1];
        if tm_queries != ref_queries
            || tm_launches[3] != launches
            || ref_launches[3] != 0
            || tm_launches[2] != tm_queries[1]
        {
            failures.push(format!(
                "{mode:?} {pass:?} {committed}+{rows}: launches {ref_launches:?} / {tm_launches:?}, queries {ref_queries:?} / {tm_queries:?}"
            ));
        }
    }
    Ok(())
}

/// The keys-token-major tile gives the block-16 tile's bits at every
/// point, bf16 and INT8 queries; a chunk of 2,048 rows straddling
/// `PRODUCT_INT8_FROM` (128 + 2,048 across 256) under the product rule
/// runs as two launches (one per query mode) on both pages.
#[test]
fn keys_token_major_tile_is_the_block16_tile() -> Result<()> {
    let Some(mut f) = fixture(
        "keys_token_major_tile_is_the_block16_tile",
        &[VLLM16, Page::KeysTokenMajor],
    )?
    else {
        return Ok(());
    };
    let mut failures = Vec::new();
    compare_points(&mut f, Queries::Bf16, &BIT_POINTS, &mut failures)?;
    if apple10() {
        let straddle_committed = PRODUCT_INT8_FROM - 128;
        let straddle = [(Pass::Prefill, straddle_committed, CHUNK)];
        let before = f.adapters[1].tile_query_launches();
        compare_points(&mut f, Queries::Product, &straddle, &mut failures)?;
        for adapter in &f.adapters {
            assert_eq!(adapter.int8_queries_from(), Some(PRODUCT_INT8_FROM));
        }
        let after = f.adapters[1].tile_query_launches();
        if [after[0] - before[0], after[1] - before[1]] != [1, 1] {
            failures.push(format!(
                "{straddle_committed}+{CHUNK} under the product rule: query launches {before:?} -> {after:?}, not one per mode"
            ));
        }
        compare_points(&mut f, Queries::Int8, &BIT_POINTS, &mut failures)?;
    } else {
        println!("ROW ktm_bits q8q not run: the device scores no INT8 queries");
    }
    assert!(
        failures.is_empty(),
        "keys-token-major tile bits: {failures:?}"
    );
    Ok(())
}

/// A tail page with 16 visible rows (its other 16 rows stored) gives
/// the block-16 bits, whose second block of that page is never read.
#[test]
fn keys_token_major_tile_masks_the_tail_page() -> Result<()> {
    let Some(mut f) = fixture(
        "keys_token_major_tile_masks_the_tail_page",
        &[VLLM16, Page::KeysTokenMajor],
    )?
    else {
        return Ok(());
    };
    for &(_, committed, rows) in &TAIL_POINTS {
        let visible = committed + rows as u32;
        assert_eq!(visible % 32, 16, "a tail page with 16 visible rows");
        assert!(f.adapters[1].current_token_count() >= visible + 16);
    }
    let mut failures = Vec::new();
    compare_points(&mut f, Queries::Bf16, &TAIL_POINTS, &mut failures)?;
    if apple10() {
        compare_points(&mut f, Queries::Int8, &TAIL_POINTS, &mut failures)?;
    }
    assert!(failures.is_empty(), "tail page bits: {failures:?}");
    Ok(())
}

/// A direct tile launch over fabricated planes: whether the validator takes
/// them. `k_shape` / `v_shape` of `dtype`, `heads` query heads over 4 KV
/// heads, `head` dimensions, `block` tokens per block, 8 verify rows at 0.
fn tile_takes(
    dtype: DType,
    k_shape: &[i64],
    v_shape: &[i64],
    heads: i64,
    head: i64,
    block: i64,
) -> Result<bool> {
    let blocks = k_shape[0];
    let int8 = dtype == DType::Int8;
    let k = MxArray::zeros(k_shape, Some(dtype))?;
    let v = MxArray::zeros(v_shape, Some(dtype))?;
    let scale_shape: Vec<i64> = if int8 {
        vec![blocks, KV_HEADS, block]
    } else {
        vec![1]
    };
    let k_scale = MxArray::zeros(&scale_shape, Some(DType::Float32))?;
    let v_scale = MxArray::zeros(&scale_shape, Some(DType::Float32))?;
    let table = MxArray::from_int32(&[0], &[1])?;
    let q = MxArray::zeros(&[VERIFY_ROWS, heads, head], Some(DType::BFloat16))?;
    let bound = prefill_plan::workspace_bound_floats(Pass::Verify, heads as u32, head as u32);
    let workspace = MxArray::zeros(&[bound as i64], Some(DType::Float32))?;
    let mut attention = std::ptr::null_mut();
    let mut workspace_out = std::ptr::null_mut();
    // SAFETY: every pointer is a live MxArray handle; the outputs are new.
    let ok = unsafe {
        mlx_sys::mlx_paged_prefill_attention_forward(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            k_scale.as_raw_ptr(),
            v_scale.as_raw_ptr(),
            table.as_raw_ptr(),
            workspace.as_raw_ptr(),
            0,
            1,
            Pass::Verify.max_splits() as i32,
            block as i32,
            heads as i32,
            KV_HEADS as i32,
            head as i32,
            if int8 { 3 } else { 1 },
            false,
            1.0 / (head as f32).sqrt(),
            &mut attention,
            &mut workspace_out,
        )
    };
    if ok {
        MxArray::from_handle(attention, "ktm_refusal_attention")?;
        MxArray::from_handle(workspace_out, "ktm_refusal_workspace")?;
    }
    Ok(ok)
}

/// The keys-token-major entry runs exactly where K is rank 4 (one
/// launch per tile launch of that page, none on the vLLM pages at block 16
/// or 32), and the validator takes a rank-4 K only as the whole INT8
/// 32-token page of heads 24/4 at D 256.
#[test]
fn tile_route_follows_the_key_rank() -> Result<()> {
    let pages = [VLLM16, VLLM32, Page::KeysTokenMajor];
    let Some(mut f) = fixture("tile_route_follows_the_key_rank", &pages)? else {
        return Ok(());
    };
    let mut modes = vec![Queries::Bf16];
    if apple10() {
        modes.push(Queries::Int8);
    }
    let mut failures = Vec::new();
    for mode in modes {
        for (adapter, page) in f.adapters.iter_mut().zip(pages) {
            set_queries(adapter, mode);
            for (pass, committed, rows) in [
                (Pass::Prefill, 1_000, CHUNK),
                (Pass::Verify, 25_000, VERIFY_ROWS),
            ] {
                let q = queries(pass, rows)?;
                let (_, launches, query_launches) = attend(adapter, &q, pass, committed)?;
                let int8 = u64::from(matches!(mode, Queries::Int8));
                let want = [
                    u64::from(pass == Pass::Prefill),
                    u64::from(pass == Pass::Verify),
                    int8,
                    u64::from(page == Page::KeysTokenMajor),
                ];
                println!(
                    "ROW ktm_route page={page:?} q={mode:?} pass={pass:?} launches={launches:?} want={want:?} queries={query_launches:?}"
                );
                if launches != want {
                    failures.push(format!(
                        "{page:?} {mode:?} {pass:?}: launches {launches:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "tile route: {failures:?}");

    let nb = 4;
    let (h, d) = (HEADS, HEAD_DIM);
    let whole = tile_takes(
        DType::Int8,
        &[nb, KV_HEADS, 32, d],
        &[nb, KV_HEADS, d, 32],
        h,
        d,
        32,
    )?;
    let refusals = [
        (
            "group 8",
            tile_takes(
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
            tile_takes(
                DType::Int8,
                &[nb, KV_HEADS, 32, 128],
                &[nb, KV_HEADS, 128, 32],
                h,
                128,
                32,
            )?,
        ),
        (
            "block 16",
            tile_takes(
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
            tile_takes(
                DType::BFloat16,
                &[nb, KV_HEADS, 32, d],
                &[nb, KV_HEADS, d, 32],
                h,
                d,
                32,
            )?,
        ),
        (
            "token-major V",
            tile_takes(
                DType::Int8,
                &[nb, KV_HEADS, 32, d],
                &[nb, KV_HEADS, 32, d],
                h,
                d,
                32,
            )?,
        ),
        (
            "rank-4 K beside a rank-5 V",
            tile_takes(
                DType::Int8,
                &[nb, KV_HEADS, 32, d],
                &[nb, KV_HEADS, d / 16, 32, 16],
                h,
                d,
                32,
            )?,
        ),
    ];
    println!("ROW ktm_validator whole={whole} refusals={refusals:?}");
    assert!(whole, "the whole keys-token-major page is refused");
    let taken: Vec<&str> = refusals
        .iter()
        .filter(|(_, ok)| *ok)
        .map(|(name, _)| *name)
        .collect();
    assert!(taken.is_empty(), "the validator took {taken:?}");
    Ok(())
}

/// Every page other families build (INT8 vLLM at block 16 and 32,
/// bf16 at block 16) keeps its product entry: no keys-token-major launch,
/// the q8q entry exactly where INT8 queries run.
#[test]
fn other_pages_keep_the_product_tile() -> Result<()> {
    let pages = [VLLM16, VLLM32, BF16_16];
    let Some(mut f) = fixture("other_pages_keep_the_product_tile", &pages)? else {
        return Ok(());
    };
    let mut failures = Vec::new();
    for (adapter, page) in f.adapters.iter_mut().zip(pages) {
        let mut modes = vec![Queries::Bf16];
        if apple10() && page.dtype() == MetalDtype::Int8 {
            modes.push(Queries::Int8);
        }
        for mode in modes {
            set_queries(adapter, mode);
            for (pass, committed, rows) in [
                (Pass::Prefill, 0, CHUNK),
                (Pass::Prefill, 65_536, CHUNK),
                (Pass::Verify, 145_584, VERIFY_ROWS),
            ] {
                let q = queries(pass, rows)?;
                let (out, launches, _) = attend(adapter, &q, pass, committed)?;
                let (_, nans, zeros, elements) = compare(&out, &out)?;
                let want = [
                    u64::from(pass == Pass::Prefill),
                    u64::from(pass == Pass::Verify),
                    u64::from(matches!(mode, Queries::Int8)),
                    0,
                ];
                println!(
                    "ROW ktm_other page={page:?} q={mode:?} pass={pass:?} committed={committed} launches={launches:?} nans={nans} zeros={zeros}"
                );
                if launches != want || nans != 0 || zeros * 100 >= elements {
                    failures.push(format!(
                        "{page:?} {mode:?} {pass:?} {committed}: launches {launches:?} (want {want:?}), nans {nans}, zeros {zeros}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "other pages: {failures:?}");
    Ok(())
}
