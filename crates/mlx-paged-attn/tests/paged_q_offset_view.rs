//! A query that is a row view at a nonzero byte offset (a slice along the
//! row axis of a larger evaluated buffer) through every paged read entry:
//! `PagedAttention`, `PagedAttentionVarlen` (bf16 v1 / v2, int8 tiled v2) and
//! `PagedPrefillAttention`. The FFI's `contiguous(q)` aliases such a view when
//! the parent buffer is within 16 KiB of it, so the primitive gets q at the
//! offset; the output must match the same rows deep-copied to offset 0 bit
//! for bit.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::prefill_plan::{self, Pass};
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::mlx_array;

struct Owned(*mut mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

const HQ: usize = 8;
const D: usize = 128;
const BS: usize = 16;

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
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

fn bits(array: &Owned) -> Vec<u32> {
    let n = unsafe { mlx_sys::mlx_array_size(array.0) };
    let mut out = vec![0f32; n];
    assert!(unsafe { mlx_sys::mlx_array_to_float32(array.0, out.as_mut_ptr(), out.len()) });
    out.iter().map(|x| x.to_bits()).collect()
}

fn eval_err(arrays: &[&Owned]) -> Option<String> {
    let mut buf = vec![0 as std::ffi::c_char; 1024];
    let mut handles: Vec<*mut mlx_array> = arrays.iter().map(|a| a.0).collect();
    let ok = unsafe {
        mlx_sys::mlx_eval_with_error(
            handles.as_mut_ptr(),
            handles.len(),
            buf.as_mut_ptr(),
            buf.len(),
        )
    };
    if ok {
        None
    } else {
        Some(
            unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Bf16,
    Int8,
}

impl Format {
    fn code(self) -> u8 {
        match self {
            Format::Bf16 => 1,
            Format::Int8 => 3,
        }
    }
}

struct Pool {
    _pool: LayerKVPool,
    format: Format,
    kv: usize,
    key: Owned,
    value: Owned,
    key_scales: Owned,
    value_scales: Owned,
    table: Vec<i32>,
    tokens: usize,
}

/// One layer of `kv` heads holding `tokens` seeded rows, logical block `l`
/// at physical block `l`.
fn filled(format: Format, kv: usize, tokens: usize) -> Option<Pool> {
    let blocks = tokens.div_ceil(BS) + 1;
    let config = PagedAttentionConfig {
        block_size: BS as u32,
        gpu_memory_mb: 512,
        head_size: D as u32,
        num_kv_heads: kv as u32,
        num_layers: 1,
        max_seq_len: Some((blocks * BS) as u32),
        max_batch_size: Some(1),
    };
    let dtype = match format {
        Format::Bf16 => MetalDtype::BFloat16,
        Format::Int8 => MetalDtype::Int8,
    };
    let pool = match LayerKVPool::new(config, blocks as u32, blocks as u32, dtype) {
        Ok(pool) => pool,
        Err(e) if e.contains("No Metal device found") => return None,
        Err(e) => panic!("LayerKVPool::new: {e}"),
    };
    let unit =
        || Owned(unsafe { mlx_sys::mlx_array_from_float32([1.0f32].as_ptr(), [1i64].as_ptr(), 1) });
    let (mut key_scales, mut value_scales) = match format {
        Format::Bf16 => (unit(), unit()),
        Format::Int8 => (
            Owned(pool.key_scale_array_raw(0).unwrap()),
            Owned(pool.value_scale_array_raw(0).unwrap()),
        ),
    };
    let mut key = Owned(pool.key_cache_array_raw(0).unwrap());
    let mut value = Owned(pool.value_cache_array_raw(0).unwrap());
    let rows: Vec<u16> = (0..tokens * kv * D)
        .map(|i| bf16(((i * 37 % 101) as f32 - 50.0) / 64.0))
        .collect();
    let shape = [tokens as i64, kv as i64, D as i64];
    let new_k = from_bf16(&rows, &shape);
    let new_v = from_bf16(&rows, &shape);
    let slots = from_i64(&(0..tokens as i64).collect::<Vec<_>>());
    let mut out = [std::ptr::null_mut::<mlx_array>(); 4];
    let [k_out, v_out, ks_out, vs_out] = &mut out;
    let ok = unsafe {
        mlx_sys::mlx_paged_kv_write_forward(
            key.0,
            value.0,
            new_k.0,
            new_v.0,
            slots.0,
            key_scales.0,
            value_scales.0,
            BS as i32,
            kv as i32,
            D as i32,
            format.code(),
            false,
            k_out,
            v_out,
            ks_out,
            vs_out,
        )
    };
    assert!(ok, "store");
    let [k, v, ks, vs] = out.map(Owned);
    assert_eq!(eval_err(&[&k, &v, &ks, &vs]), None);
    (key, value, key_scales, value_scales) = (k, v, ks, vs);
    Some(Pool {
        _pool: pool,
        format,
        kv,
        key,
        value,
        key_scales,
        value_scales,
        table: (0..blocks as i32 - 1).collect(),
        tokens,
    })
}

/// Rows `1..=rows` of an evaluated `[rows + 1, HQ, D]` bf16 buffer: a view
/// at offset one row (`HQ * D * 2` bytes), and the same rows deep-copied.
fn view_and_copy(rows: usize) -> (Owned, Owned, Owned) {
    let data: Vec<u16> = (0..(rows + 1) * HQ * D)
        .map(|i| bf16(((i * 13 % 29) as f32 - 14.0) / 16.0))
        .collect();
    let parent = from_bf16(&data, &[(rows + 1) as i64, HQ as i64, D as i64]);
    assert_eq!(eval_err(&[&parent]), None);
    let view = Owned(unsafe { mlx_sys::mlx_array_slice_axis(parent.0, 0, 1, (rows + 1) as i64) });
    let copy = Owned(unsafe { mlx_sys::mlx_array_deep_copy(view.0) });
    assert_eq!(eval_err(&[&copy]), None);
    (parent, view, copy)
}

fn decode(pool: &Pool, q: &Owned) -> Owned {
    let table = from_i32(&pool.table, &[1, pool.table.len() as i64]);
    let lens = from_i32(&[pool.tokens as i32], &[1]);
    let out = Owned(unsafe {
        mlx_sys::mlx_paged_attention_forward(
            q.0,
            pool.key.0,
            pool.value.0,
            table.0,
            lens.0,
            pool.key_scales.0,
            pool.value_scales.0,
            1.0 / (D as f32).sqrt(),
            0.0,
            0,
            BS as i32,
            HQ as i32,
            pool.kv as i32,
            D as i32,
            pool.format.code(),
        )
    });
    assert!(!out.0.is_null(), "decode factory refused");
    out
}

fn varlen(pool: &Pool, q: &Owned, rows: usize) -> Owned {
    let table = from_i32(&pool.table, &[1, pool.table.len() as i64]);
    let lens = from_i32(&[pool.tokens as i32], &[1]);
    let cu = from_i32(&[0, rows as i32], &[2]);
    let out = Owned(unsafe {
        mlx_sys::mlx_paged_attention_varlen_forward(
            q.0,
            pool.key.0,
            pool.value.0,
            table.0,
            lens.0,
            cu.0,
            pool.key_scales.0,
            pool.value_scales.0,
            1.0 / (D as f32).sqrt(),
            0.0,
            0,
            BS as i32,
            HQ as i32,
            pool.kv as i32,
            D as i32,
            pool.format.code(),
        )
    });
    assert!(!out.0.is_null(), "varlen factory refused");
    out
}

fn prefill(pool: &Pool, q: &Owned, rows: usize) -> Owned {
    let committed = pool.tokens - rows;
    let splits = prefill_plan::splits(Pass::Prefill, rows as u32, committed as u32, 32);
    let floats = prefill_plan::workspace_floats(rows as u32, splits, HQ as u32, D as u32);
    let workspace = Owned(unsafe { mlx_sys::mlx_array_zeros([floats as i64].as_ptr(), 1, 0) });
    let table = from_i32(&pool.table, &[pool.table.len() as i64]);
    let mut attention = std::ptr::null_mut();
    let mut workspace_out = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_paged_prefill_attention_forward(
            q.0,
            pool.key.0,
            pool.value.0,
            pool.key_scales.0,
            pool.value_scales.0,
            table.0,
            workspace.0,
            committed as i32,
            splits as i32,
            Pass::Prefill.max_splits() as i32,
            BS as i32,
            HQ as i32,
            pool.kv as i32,
            D as i32,
            pool.format.code(),
            1.0 / (D as f32).sqrt(),
            &mut attention,
            &mut workspace_out,
        )
    };
    assert!(ok, "prefill factory refused");
    drop(Owned(workspace_out));
    Owned(attention)
}

/// The op on the view evaluates and matches the op on the copy bit for bit.
fn assert_view_matches_copy(name: &str, rows: usize, op: impl Fn(&Owned) -> Owned) {
    let (_parent, view, copy) = view_and_copy(rows);
    let got = op(&view);
    if let Some(err) = eval_err(&[&got]) {
        panic!("{name}: q view at a nonzero offset refused: {err}");
    }
    let expected = op(&copy);
    assert_eq!(eval_err(&[&expected]), None, "{name}: offset-0 copy");
    let (got, expected) = (bits(&got), bits(&expected));
    let differing = got.iter().zip(&expected).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{name}: view differs from the copy");
}

/// Decode (`PagedAttention`) over 40 tokens (v1) and 1,500 (v2).
#[test]
fn decode_reads_a_q_view_at_an_offset() {
    for tokens in [40, 1500] {
        let Some(pool) = filled(Format::Bf16, 2, tokens) else {
            return;
        };
        assert_view_matches_copy(&format!("decode c={tokens}"), 1, |q| decode(&pool, q));
    }
}

/// Varlen (`PagedAttentionVarlen`) over bf16 v1 / v2, 2 rows (the grouped
/// selector's row count) and 3.
#[test]
fn varlen_reads_a_q_view_at_an_offset() {
    for tokens in [40, 1500] {
        let Some(pool) = filled(Format::Bf16, 2, tokens) else {
            return;
        };
        for rows in [2, 3] {
            assert_view_matches_copy(&format!("varlen c={tokens} rows={rows}"), rows, |q| {
                varlen(&pool, q, rows)
            });
        }
    }
}

/// Int8 varlen of one sequence runs in row tiles, each binding q at its own
/// tile offset on top of the view's: 40,000 tokens are 79 partitions, so
/// 30 rows are 2 tiles of 25 + 5.
#[test]
fn int8_tiled_varlen_reads_a_q_view_at_an_offset() {
    let Some(pool) = filled(Format::Int8, 2, 40_000) else {
        return;
    };
    assert_view_matches_copy("int8 varlen tiled", 30, |q| varlen(&pool, q, 30));
}

/// Paged prefill with one KV head and 32 rows: the staging transpose only
/// moves a size-1 axis and needs no pad, so the staged q is the view itself.
#[test]
fn prefill_reads_a_staged_q_view_at_an_offset() {
    // SAFETY: nullary predicate that catches internally.
    if unsafe { mlx_sys::mlx_test_gpu_lacks_apple_family() } {
        eprintln!("skipping: no Apple GPU family; the paged prefill kernels need Apple7+");
        return;
    }
    let Some(pool) = filled(Format::Bf16, 1, 1024 + 32) else {
        return;
    };
    assert_view_matches_copy("prefill", 32, |q| prefill(&pool, q, 32));
}

/// A q view whose offset is not whole heads is still refused: one bf16
/// element past the start of a flat buffer, reshaped to `[1, HQ, D]`.
#[test]
fn a_q_view_off_a_head_boundary_is_refused() {
    let Some(pool) = filled(Format::Bf16, 2, 40) else {
        return;
    };
    let n = HQ * D;
    let data: Vec<u16> = (0..n + 1).map(|i| bf16((i % 7) as f32 / 8.0)).collect();
    let flat = from_bf16(&data, &[(n + 1) as i64]);
    assert_eq!(eval_err(&[&flat]), None);
    let view = Owned(unsafe { mlx_sys::mlx_array_slice_axis(flat.0, 0, 1, (n + 1) as i64) });
    let q =
        Owned(unsafe { mlx_sys::mlx_array_reshape(view.0, [1, HQ as i64, D as i64].as_ptr(), 3) });
    let out = decode(&pool, &q);
    let err = eval_err(&[&out]).expect("a q offset of 2 bytes must be refused");
    assert!(
        err.contains("q offset must be a multiple of one head"),
        "unexpected error: {err}"
    );
}
