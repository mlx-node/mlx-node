//! The paged pool page for the 12 indexed layers: the 576-wide f16 latent row
//! split over the K plane (cols 0..288) and the V plane (cols 288..576) of a
//! token-major head-288 page, and the pooled indexer keys in an f16 index
//! plane of 8 x 128 per block, against `pool-layout.json`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PageLayout, PagedAttentionConfig};
use serde_json::Value;
use sha2::Digest;

use crate::array::{DType, MxArray};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

const BS: usize = 32;
const HALF: usize = 288;
const ROW: usize = 576;
const DK: usize = 128;
const INDEX_ROWS: usize = 8;
const BLOCKS: u32 = 12;
const TOKENS: usize = 70;
const GROUPS: usize = TOKENS / 4;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash/pool-layout.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn config(num_layers: u32) -> PagedAttentionConfig {
    PagedAttentionConfig {
        block_size: BS as u32,
        gpu_memory_mb: 256,
        head_size: HALF as u32,
        num_kv_heads: 1,
        num_layers,
        max_seq_len: Some(1024),
        max_batch_size: Some(4),
    }
}

fn layout() -> PageLayout {
    PageLayout {
        values_token_major: true,
        keys_token_major: false,
        index_rows: INDEX_ROWS as u32,
        index_dimension: DK as u32,
    }
}

/// A Shared f16 pool with an f16 index plane, `None` without Metal.
fn pool(num_layers: u32) -> Option<Arc<LayerKVPool>> {
    match LayerKVPool::new_with_index_dtype(
        config(num_layers),
        BLOCKS,
        BLOCKS,
        MetalDtype::Float16,
        layout(),
        MetalDtype::Float16,
    ) {
        Ok(pool) => Some(Arc::new(pool)),
        Err(error) if crate::test_support::metal_device_absent(&error.to_string()) => {
            eprintln!("skipping Hy4 pool test: {error}");
            None
        }
        Err(error) => panic!("Hy4 pool: {error}"),
    }
}

fn row_bits(t: usize, c: usize) -> u16 {
    0x3C00 | ((t * 37 + c * 11) % 1024) as u16
}

fn pooled_bits(g: usize, c: usize) -> u16 {
    0x3800 | ((g * 13 + c * 5) % 1024) as u16
}

fn sha_u16(bits: &[u16]) -> String {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    sha2::Sha256::digest(&bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn f16_bits(array: &MxArray) -> Vec<u16> {
    assert_eq!(array.dtype().unwrap(), DType::Float16);
    array.to_uint16_native().unwrap()
}

fn usizes(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn slot(table: &[usize], t: usize) -> usize {
    table[t / BS] * BS + t % BS
}

/// `[tokens, 1, 288]` f16 rows of columns `cols` of the synthetic latent.
fn half_rows(cols: std::ops::Range<usize>) -> MxArray {
    let bits: Vec<u16> = (0..TOKENS)
        .flat_map(|t| cols.clone().map(move |c| row_bits(t, c)))
        .collect();
    MxArray::from_float16(&bits, &[TOKENS as i64, 1, HALF as i64]).unwrap()
}

fn pooled_rows() -> MxArray {
    let bits: Vec<u16> = (0..GROUPS)
        .flat_map(|g| (0..DK).map(move |c| pooled_bits(g, c)))
        .collect();
    MxArray::from_float16(&bits, &[GROUPS as i64, DK as i64]).unwrap()
}

/// The pool takes the Hy4 geometry: head 288 passes the config check, the
/// K/V bytes per block are the paged sizer's, the index slot is 2,048 B, and
/// the pool's block is 466,944 B with a (K, V + index) host blob.
#[test]
fn pool_geometry_matches_the_fixture() {
    let fixture = fixture();
    let bytes = &fixture["bytes"];
    let want = |key: &str| bytes[key].as_u64().unwrap();
    config(12).validate().unwrap();
    assert_eq!(
        mlx_paged_attn::profile::bytes_per_block(
            12,
            1,
            HALF as u32,
            BS as u32,
            MetalDtype::Float16
        ),
        Ok(want("pa_bytes_per_block_no_index"))
    );
    assert_eq!(layout().index_bytes(), want("page_layout_index_bytes"));
    let Some(pool) = pool(12) else {
        return;
    };
    assert_eq!(pool.bytes_per_block(), Ok(want("per_block_all_layers")));
    assert_eq!(
        pool.block_blob_bytes(),
        Ok((
            want("key_plane_per_block_layer"),
            want("value_plane_per_block_layer") + want("index_plane_per_block_layer")
        ))
    );
    assert_eq!(pool.index_dtype(), MetalDtype::Float16);
}

/// One f16 token-major store writes the two half rows at their slots: the K
/// and V planes hash as the fixture's, and `take K ‖ take V` at the gather
/// cells (−1 read as slot 0) gives the fixture's rows, the valid ones equal
/// to the flat latent rows.
/// Mutation: the store writes the vLLM page (`values_token_major` false) →
/// red; the f16 `_vt` instance missing → the write fails.
#[test]
fn half_rows_land_in_the_planes_by_slot() {
    let Some(pool) = pool(1) else {
        return;
    };
    let fixture = fixture();
    let synthetic = &fixture["synthetic"];
    let table = usizes(&synthetic["block_table"]);
    let slots: Vec<i64> = (0..TOKENS).map(|t| slot(&table, t) as i64).collect();
    let slots = MxArray::from_int64(&slots, &[TOKENS as i64]).unwrap();
    let keys = half_rows(0..HALF);
    let values = half_rows(HALF..ROW);
    let k_pool = MxArray::from_handle(pool.key_cache_array_raw(0).unwrap(), "k pool").unwrap();
    let v_pool = MxArray::from_handle(pool.value_cache_array_raw(0).unwrap(), "v pool").unwrap();
    let unit = MxArray::from_float32(&[1.0], &[1]).unwrap();
    let mut out = [std::ptr::null_mut(); 4];
    let [ko, vo, kso, vso] = &mut out;
    // SAFETY: every input is a live MLX array handle; each output is written
    // only on success and is then owned here.
    let ok = unsafe {
        mlx_sys::mlx_paged_kv_write_forward(
            k_pool.as_raw_ptr(),
            v_pool.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            slots.as_raw_ptr(),
            unit.as_raw_ptr(),
            unit.as_raw_ptr(),
            BS as i32,
            1,
            HALF as i32,
            0,
            true,
            ko,
            vo,
            kso,
            vso,
        )
    };
    assert!(ok, "f16 token-major store");
    let [k, v, ks, vs] = out.map(|raw| MxArray::from_handle(raw, "store output").unwrap());
    drop((ks, vs));
    assert_eq!(
        k.shape().unwrap().to_vec(),
        vec![i64::from(BLOCKS), 1, BS as i64, HALF as i64]
    );
    assert_eq!(
        sha_u16(&f16_bits(&k)),
        synthetic["k_plane_sha256_u16le"].as_str().unwrap()
    );
    assert_eq!(
        sha_u16(&f16_bits(&v)),
        synthetic["v_plane_sha256_u16le"].as_str().unwrap()
    );

    let gather = &synthetic["gather"];
    let cells: Vec<i64> = gather["cells"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_i64().unwrap())
        .collect();
    let gid: Vec<i32> = cells
        .iter()
        .map(|&t| usize::try_from(t).map_or(0, |t| slot(&table, t) as i32))
        .collect();
    let gid = MxArray::from_int32(&gid, &[cells.len() as i64]).unwrap();
    let rows = i64::from(BLOCKS) * BS as i64;
    let k_rows = k
        .reshape(&[rows, HALF as i64])
        .unwrap()
        .take(&gid, 0)
        .unwrap();
    let v_rows = v
        .reshape(&[rows, HALF as i64])
        .unwrap()
        .take(&gid, 0)
        .unwrap();
    let gathered = f16_bits(&MxArray::concatenate(&k_rows, &v_rows, 1).unwrap());
    assert_eq!(
        sha_u16(&gathered),
        gather["all_rows_sha256_u16le"].as_str().unwrap()
    );
    let mut valid = Vec::new();
    for (i, &t) in cells.iter().enumerate() {
        let Ok(t) = usize::try_from(t) else {
            continue;
        };
        let row = &gathered[i * ROW..(i + 1) * ROW];
        let flat: Vec<u16> = (0..ROW).map(|c| row_bits(t, c)).collect();
        assert_eq!(row, flat.as_slice(), "cell {i} (token {t})");
        valid.extend_from_slice(row);
    }
    assert_eq!(
        sha_u16(&valid),
        gather["valid_rows_sha256_u16le"].as_str().unwrap()
    );
}

/// The pooled keys of the 17 closed groups go to their f16 index plane rows
/// (`physical_block · 8 + group % 8`) and the plane hashes as the fixture's;
/// bf16 rows into the f16 plane are refused.
/// Mutation: the store output keeps the bf16 dtype → red.
#[test]
fn pooled_keys_land_in_the_f16_index_plane() {
    let Some(pool) = pool(1) else {
        return;
    };
    let fixture = fixture();
    let index = &fixture["synthetic"]["index_plane"];
    let slots: Vec<i32> = index["placement"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["index_slot"].as_i64().unwrap() as i32)
        .collect();
    assert_eq!(slots.len(), GROUPS);
    let slots = MxArray::from_int32(&slots, &[GROUPS as i64]).unwrap();
    let plane = MxArray::from_handle(pool.index_array_raw(0).unwrap(), "index plane").unwrap();
    assert_eq!(plane.dtype().unwrap(), DType::Float16);
    let store = |rows: &MxArray| -> Option<MxArray> {
        let mut out = std::ptr::null_mut();
        // SAFETY: every input is a live MLX array handle; `out` is written
        // only on success and is then owned here.
        let ok = unsafe {
            mlx_sys::mlx_paged_index_store(
                plane.as_raw_ptr(),
                rows.as_raw_ptr(),
                slots.as_raw_ptr(),
                &mut out,
            )
        };
        ok.then(|| MxArray::from_handle(out, "index store").unwrap())
    };
    let rows = pooled_rows();
    assert!(store(&rows.astype(DType::BFloat16).unwrap()).is_none());
    let written = store(&rows).expect("f16 index store");
    assert_eq!(
        written.shape().unwrap().to_vec(),
        vec![i64::from(BLOCKS), INDEX_ROWS as i64, DK as i64]
    );
    assert_eq!(
        sha_u16(&f16_bits(&written)),
        index["sha256_u16le"].as_str().unwrap()
    );
}

/// The adapter stores pooled keys in the pool's index dtype: f32 rows come
/// back from `index_rows` as the same f16 values, for the 17 closed groups
/// of a 70-token request.
/// Mutation: `store_index_rows` casts to bf16 → the store is refused → red.
#[test]
fn adapter_stores_pooled_keys_in_the_index_dtype() {
    let Some(pool) = pool(1) else {
        return;
    };
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(BLOCKS, BLOCKS, BS as u32)));
    let mut adapter = PagedKVCacheAdapter::new(allocator, pool, BS as u32).unwrap();
    adapter.begin_request(1).unwrap();
    adapter.record_tokens(&[3; TOKENS]).unwrap();
    let rows = pooled_rows();
    adapter
        .store_index_rows(0, &rows.astype(DType::Float32).unwrap(), 0)
        .unwrap();
    let back = adapter.index_rows(0, GROUPS as u32).unwrap();
    assert_eq!(
        back.shape().unwrap().to_vec(),
        vec![GROUPS as i64, DK as i64]
    );
    assert_eq!(f16_bits(&back), f16_bits(&rows));
}
