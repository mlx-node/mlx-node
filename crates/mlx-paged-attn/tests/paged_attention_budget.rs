//! The paged read charges its command buffer for the pool blocks its tables
//! reach, not the whole pool (Splash `EncoderResidency`: memory no command
//! uses costs no command buffer anything). A pool far above the per-buffer
//! size limit with one live block must not force a size commit per read.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::mlx_array;

const H: usize = 4;
const HQ: usize = 24;
const D: usize = 256;
const BS: usize = 16;
/// 4,096 blocks of 16,384 K (and V) elements: each plane charges 64 Mi
/// units, above every device's 40-50 MB per-command-buffer limit.
const BLOCKS: u32 = 4096;
const READS: i32 = 8;

struct Owned(*mut mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

fn from_bf16(data: &[u16], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_bfloat16(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_i32(data: &[i32], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_int32(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_i64(data: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_int64(data.as_ptr(), [data.len() as i64].as_ptr(), 1) })
}

fn unit_scale() -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_float32([1.0f32].as_ptr(), [1i64].as_ptr(), 1) })
}

/// bf16 bits of a small deterministic ramp.
fn ramp(len: usize, step: f32) -> Vec<u16> {
    (0..len)
        .map(|i| ((((i % 251) as f32 - 125.0) * step).to_bits() >> 16) as u16)
        .collect()
}

/// A one-layer pool of `BLOCKS` blocks whose block 0 holds 16 written
/// tokens, as the four arrays a read binds.
struct Pool {
    _pool: LayerKVPool,
    key: Owned,
    value: Owned,
    key_scale: Owned,
    value_scale: Owned,
    kv_dtype: u8,
}

fn pool(cache_dtype: MetalDtype) -> Option<Pool> {
    let config = PagedAttentionConfig {
        block_size: BS as u32,
        gpu_memory_mb: 4096,
        head_size: D as u32,
        num_kv_heads: H as u32,
        num_layers: 1,
        max_seq_len: Some(BLOCKS * BS as u32),
        max_batch_size: Some(1),
    };
    let pool = match LayerKVPool::new(config, BLOCKS, BLOCKS, cache_dtype) {
        Ok(pool) => pool,
        Err(e) if e.contains("No Metal device found") => return None,
        Err(e) => panic!("LayerKVPool::new: {e}"),
    };
    let int8 = cache_dtype == MetalDtype::Int8;
    let kv_dtype = if int8 { 3 } else { 1 };
    let key = Owned(pool.key_cache_array_raw(0).unwrap());
    let value = Owned(pool.value_cache_array_raw(0).unwrap());
    let (key_scale, value_scale) = if int8 {
        (
            Owned(pool.key_scale_array_raw(0).unwrap()),
            Owned(pool.value_scale_array_raw(0).unwrap()),
        )
    } else {
        (unit_scale(), unit_scale())
    };
    let shape = [BS as i64, H as i64, D as i64];
    let new_k = from_bf16(&ramp(BS * H * D, 0.01), &shape);
    let new_v = from_bf16(&ramp(BS * H * D, 0.02), &shape);
    let slots = from_i64(&(0..BS as i64).collect::<Vec<_>>());
    let mut out = [std::ptr::null_mut::<mlx_array>(); 4];
    let [k_out, v_out, ks_out, vs_out] = &mut out;
    let ok = unsafe {
        mlx_sys::mlx_paged_kv_write_forward(
            key.0,
            value.0,
            new_k.0,
            new_v.0,
            slots.0,
            key_scale.0,
            value_scale.0,
            BS as i32,
            H as i32,
            D as i32,
            kv_dtype,
            k_out,
            v_out,
            ks_out,
            vs_out,
        )
    };
    assert!(ok, "store into the {cache_dtype:?} pool");
    let written = out.map(Owned);
    let mut handles = out;
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 4) });
    let [key, value, key_scale, value_scale] = written;
    Some(Pool {
        _pool: pool,
        key,
        value,
        key_scale,
        value_scale,
        kv_dtype,
    })
}

/// Size commits of `READS` one-row reads of block 0 (seq_len 16).
fn size_commits(pool: &Pool, varlen: bool, commit_each: bool) -> i64 {
    let table = from_i32(&[0], &[1, 1]);
    let lens = from_i32(&[BS as i32], &[1]);
    reads(pool, &table, &lens, 1, varlen, commit_each)
}

fn reads(
    pool: &Pool,
    table: &Owned,
    lens: &Owned,
    rows: usize,
    varlen: bool,
    commit_each: bool,
) -> i64 {
    let q = from_bf16(
        &ramp(rows * HQ * D, 0.003),
        &[rows as i64, HQ as i64, D as i64],
    );
    let cu: Vec<i32> = (0..=rows as i32).collect();
    let cu = from_i32(&cu, &[rows as i64 + 1]);
    let count = unsafe {
        mlx_sys::mlx_test_paged_attention_commit_count(
            q.0,
            pool.key.0,
            pool.value.0,
            pool.key_scale.0,
            pool.value_scale.0,
            table.0,
            lens.0,
            if varlen { cu.0 } else { std::ptr::null_mut() },
            pool.kv_dtype,
            READS,
            commit_each,
        )
    };
    assert!(count >= 0, "commit-count hook failed");
    count
}

#[test]
fn budgeted_pool_bind_adds_no_commit() {
    let mut failures = Vec::new();
    for cache_dtype in [MetalDtype::Int8, MetalDtype::BFloat16] {
        let Some(pool) = pool(cache_dtype) else {
            return;
        };
        for varlen in [false, true] {
            let alone = size_commits(&pool, varlen, true);
            let shared = size_commits(&pool, varlen, false);
            eprintln!(
                "{cache_dtype:?} varlen={varlen}: {READS} reads, one per command buffer: \
                 {alone} size commits; all in one command buffer: {shared}"
            );
            // A read alone in its command buffer charges one live block.
            if alone != 0 {
                failures.push(format!(
                    "{cache_dtype:?} varlen={varlen}: {alone} size commits for {READS} reads \
                     each alone in its command buffer, expected 0"
                ));
            }
            // MLX fork patch 10 restores the full charge when a command
            // buffer binds the same pool again, so every second read commits.
            if shared != i64::from(READS / 2) {
                failures.push(format!(
                    "{cache_dtype:?} varlen={varlen}: {shared} size commits for {READS} reads in \
                     one command buffer, expected {}",
                    READS / 2
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Two rows over the same 1,000 blocks charge them once (Splash
/// `PageStorage::writeTable` dedups the entries it records): 1,000 K and V
/// blocks stay under the per-buffer limit, 2,000 do not.
#[test]
fn a_shared_prefix_is_charged_once() {
    const SHARED: usize = 1000;
    let Some(pool) = pool(MetalDtype::Int8) else {
        return;
    };
    let row: Vec<i32> = (0..SHARED as i32).collect();
    let table = from_i32(&[row.clone(), row].concat(), &[2, SHARED as i64]);
    let len = (SHARED * BS) as i32;
    let lens = from_i32(&[len, len], &[2]);
    let commits = reads(&pool, &table, &lens, 2, false, true);
    assert_eq!(
        commits, 0,
        "two rows sharing {SHARED} blocks: {commits} size commits over {READS} reads"
    );
}
