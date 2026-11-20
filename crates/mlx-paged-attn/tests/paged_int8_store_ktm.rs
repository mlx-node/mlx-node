//! Int8 store on the keys-token-major page (K `[nb, H, 32, D]` token-major,
//! V `[nb, H, D, 32]` dimension-major) through `mlx_paged_kv_write_forward`,
//! against the vLLM page store of the same rows, the f32 reference, and the
//! write validator.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{LayerKVPool, PageLayout, PagedAttentionConfig};
use mlx_sys::mlx_array;

const H: usize = 4;
const X: usize = 16;
const INT8: u8 = 3;
const BF16: u8 = 1;

struct Owned(*mut mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Page {
    Vllm,
    KeysTokenMajor,
    TokenMajor,
}

impl Page {
    fn layout(self) -> PageLayout {
        match self {
            Page::Vllm => PageLayout::default(),
            Page::KeysTokenMajor => PageLayout {
                keys_token_major: true,
                ..PageLayout::default()
            },
            Page::TokenMajor => PageLayout {
                values_token_major: true,
                ..PageLayout::default()
            },
        }
    }
}

fn pool(
    page: Page,
    head_size: usize,
    block_size: usize,
    num_blocks: usize,
    dtype: MetalDtype,
) -> Option<LayerKVPool> {
    let config = PagedAttentionConfig {
        block_size: block_size as u32,
        gpu_memory_mb: 256,
        head_size: head_size as u32,
        num_kv_heads: H as u32,
        num_layers: 1,
        max_seq_len: Some((num_blocks * block_size) as u32),
        max_batch_size: Some(1),
    };
    match LayerKVPool::new_with_layout(
        config,
        num_blocks as u32,
        num_blocks as u32,
        dtype,
        page.layout(),
    ) {
        Ok(pool) => Some(pool),
        Err(e) if e.contains("No Metal device found") => None,
        Err(e) => panic!("LayerKVPool::new_with_layout {page:?}: {e}"),
    }
}

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn f32_of(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn from_bf16(data: &[u16], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_bfloat16(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_f32(data: &[f32], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_float32(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_i64(data: &[i64]) -> Owned {
    let shape = [data.len() as i64];
    Owned(unsafe { mlx_sys::mlx_array_from_int64(data.as_ptr(), shape.as_ptr(), 1) })
}

/// The pool arrays one write binds: K, V, K scales, V scales.
struct Planes {
    key: Owned,
    value: Owned,
    key_scales: Owned,
    value_scales: Owned,
}

fn planes(pool: &LayerKVPool) -> Planes {
    let int8 = pool.cache_dtype() == MetalDtype::Int8;
    let scale = |key: bool| {
        if !int8 {
            return from_f32(&[1.0], &[1]);
        }
        Owned(if key {
            pool.key_scale_array_raw(0).unwrap()
        } else {
            pool.value_scale_array_raw(0).unwrap()
        })
    };
    Planes {
        key: Owned(pool.key_cache_array_raw(0).unwrap()),
        value: Owned(pool.value_cache_array_raw(0).unwrap()),
        key_scales: scale(true),
        value_scales: scale(false),
    }
}

/// One `mlx_paged_kv_write_forward` over the given arrays, evaluated.
/// `false` when the bridge refused it.
#[allow(clippy::too_many_arguments)]
fn write_arrays(
    key: &Owned,
    value: &Owned,
    key_scales: &Owned,
    value_scales: &Owned,
    rows: (&[u16], &[u16]),
    slots: &[i64],
    block_size: usize,
    head_size: usize,
    kv_dtype: u8,
    values_token_major: bool,
) -> bool {
    let shape = [slots.len() as i64, H as i64, head_size as i64];
    let new_k = from_bf16(rows.0, &shape);
    let new_v = from_bf16(rows.1, &shape);
    let slots = from_i64(slots);
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
            block_size as i32,
            H as i32,
            head_size as i32,
            kv_dtype,
            values_token_major,
            k_out,
            v_out,
            ks_out,
            vs_out,
        )
    };
    let outputs: Vec<Owned> = out.iter().map(|&raw| Owned(raw)).collect();
    if !ok {
        return false;
    }
    let mut handles = out;
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 4) });
    drop(outputs);
    true
}

fn write(pool: &LayerKVPool, rows: (&[u16], &[u16]), slots: &[i64], page: Page) {
    let config = pool.config();
    let planes = planes(pool);
    assert!(
        write_arrays(
            &planes.key,
            &planes.value,
            &planes.key_scales,
            &planes.value_scales,
            rows,
            slots,
            config.block_size as usize,
            config.head_size as usize,
            INT8,
            page == Page::TokenMajor,
        ),
        "{page:?} write refused"
    );
}

/// Every block's host blob set to `byte`; returns the blob pair.
fn fill(pool: &LayerKVPool, blocks: usize, byte: u8) -> (Vec<u8>, Vec<u8>) {
    let (k, v) = pool.block_blob_bytes().unwrap();
    let (k, v) = (vec![byte; k as usize], vec![byte; v as usize]);
    let uploads: Vec<_> = (0..blocks as u32)
        .map(|block| (block, vec![(k.as_slice(), v.as_slice())]))
        .collect();
    pool.write_blocks_all_layers(&uploads).unwrap();
    (k, v)
}

/// Splitmix64: deterministic rows with no crate dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// `[T, H, D]` bf16 rows at several magnitudes, the row (token 2, head 1)
/// all zero.
fn rows(tokens: usize, head_size: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    let mut out = vec![0u16; tokens * H * head_size];
    for t in 0..tokens {
        for h in 0..H {
            if (t, h) == (2, 1) {
                continue;
            }
            let magnitude = [0.01f32, 1.0, 3.7, 40.0, 250.0][(t * H + h) % 5];
            for d in 0..head_size {
                out[(t * H + h) * head_size + d] = bf16(rng.next() * magnitude);
            }
        }
    }
    out
}

/// Splash `splash_store_kv_row` in f32, in its order: `(v * 127) / max`.
fn quantize(row: &[f32]) -> (Vec<i8>, f32) {
    let maximum = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    if maximum == 0.0 {
        return (vec![0; row.len()], 0.0);
    }
    let q = row
        .iter()
        .map(|v| {
            ((v * 127.0f32) / maximum)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8
        })
        .collect();
    (q, maximum / 127.0f32)
}

/// Offset of (head, page token, dimension) inside one block's K or V data.
fn element(
    page: Page,
    key: bool,
    bs: usize,
    head_size: usize,
    h: usize,
    t: usize,
    d: usize,
) -> usize {
    let token_major = page == Page::TokenMajor || (key && page == Page::KeysTokenMajor);
    if token_major {
        (h * bs + t) * head_size + d
    } else if key {
        ((h * (head_size / X) + d / X) * bs + t) * X + d % X
    } else {
        (h * head_size + d) * bs + t
    }
}

/// The blobs every block holds after `fill(.., sentinel)` and a write of
/// `keys` / `values` at `slots`, from the f32 reference.
#[allow(clippy::too_many_arguments)]
fn expected(
    page: Page,
    blank: &(Vec<u8>, Vec<u8>),
    blocks: usize,
    bs: usize,
    head_size: usize,
    keys: &[u16],
    values: &[u16],
    slots: &[i64],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let data = H * head_size * bs;
    let mut out = vec![blank.clone(); blocks];
    for (t, &slot) in slots.iter().enumerate() {
        if slot < 0 {
            continue;
        }
        let (block, token) = (slot as usize / bs, slot as usize % bs);
        for h in 0..H {
            let start = (t * H + h) * head_size;
            for (side, source) in [keys, values].into_iter().enumerate() {
                let row: Vec<f32> = source[start..start + head_size]
                    .iter()
                    .map(|&b| f32_of(b))
                    .collect();
                let (q, scale) = quantize(&row);
                let blob = if side == 0 {
                    &mut out[block].0
                } else {
                    &mut out[block].1
                };
                for (d, &q) in q.iter().enumerate() {
                    blob[element(page, side == 0, bs, head_size, h, token, d)] = q as u8;
                }
                let at = data + (h * bs + token) * 4;
                blob[at..at + 4].copy_from_slice(&scale.to_ne_bytes());
            }
        }
    }
    out
}

/// Slots over every token of blocks `order`, out of order, both halves of a
/// 32-token block, with one skipped token (slot -1).
fn slots(tokens: usize, bs: usize, order: &[usize]) -> Vec<i64> {
    (0..tokens)
        .map(|t| {
            if t == 7 {
                -1
            } else {
                let block = order[t % order.len()];
                let token = (t / order.len() * 5) % bs;
                (block * bs + token) as i64
            }
        })
        .collect()
}

/// The new page's store writes the vLLM page store's bytes for the same
/// rows with K transposed out of the x = 16 pack: V and both scale planes
/// byte-equal, K `[H, 32, D]` = vLLM K `[H, D/16, 32, 16]` element for
/// element. Mutation: the K halves of a page swapped -> red at the K assert.
#[test]
fn keys_token_major_store_is_the_transposed_vllm_store() {
    const BS: usize = 32;
    const BLOCKS: usize = 6;
    const SENTINEL: u8 = 0x5a;
    for head_size in [128usize, 256] {
        let Some(vllm) = pool(Page::Vllm, head_size, BS, BLOCKS, MetalDtype::Int8) else {
            return;
        };
        let Some(ktm) = pool(
            Page::KeysTokenMajor,
            head_size,
            BS,
            BLOCKS,
            MetalDtype::Int8,
        ) else {
            return;
        };
        assert_eq!(vllm.block_blob_bytes(), ktm.block_blob_bytes());
        assert_eq!(vllm.bytes_per_block(), ktm.bytes_per_block());
        let tokens = 128;
        let slots = slots(tokens, BS, &[5, 0, 2, 3]);
        let keys = rows(tokens, head_size, 0x5eed + head_size as u64);
        let values = rows(tokens, head_size, 0xfeed + head_size as u64);
        let blank = fill(&vllm, BLOCKS, SENTINEL);
        assert_eq!(fill(&ktm, BLOCKS, SENTINEL), blank);
        write(&vllm, (&keys, &values), &slots, Page::Vllm);
        write(&ktm, (&keys, &values), &slots, Page::KeysTokenMajor);

        let data = H * head_size * BS;
        let mut written = 0usize;
        let mut key_mismatch = 0usize;
        for block in 0..BLOCKS {
            let want = &vllm.read_block_all_layers(block as u32).unwrap()[0];
            let got = &ktm.read_block_all_layers(block as u32).unwrap()[0];
            written += want.0[..data].iter().filter(|&&b| b != SENTINEL).count();
            assert!(
                got.0[data..] == want.0[data..],
                "D {head_size} block {block}: K scales"
            );
            assert!(
                got.1[..data] == want.1[..data],
                "D {head_size} block {block}: V"
            );
            assert!(
                got.1[data..] == want.1[data..],
                "D {head_size} block {block}: V scales"
            );
            for h in 0..H {
                for t in 0..BS {
                    for d in 0..head_size {
                        let new =
                            got.0[element(Page::KeysTokenMajor, true, BS, head_size, h, t, d)];
                        let old = want.0[element(Page::Vllm, true, BS, head_size, h, t, d)];
                        key_mismatch += usize::from(new != old);
                    }
                }
            }
        }
        assert!(
            written > tokens * H * head_size / 2,
            "D {head_size}: the write reached the pool ({written} K bytes)"
        );
        assert_eq!(
            key_mismatch, 0,
            "D {head_size}: K is not the transposed vLLM K"
        );
    }
}

/// The vLLM page keeps its store at block 16 and 32, and the token-major
/// page (`values_token_major`) keeps its own: every byte equals the f32
/// reference at the page's index formulas. Mutations: the new store taken
/// for a rank-5 K (vLLM page red), the new store taken on the K rank alone
/// (token-major page red).
#[test]
fn other_pages_keep_their_store() {
    const BLOCKS: usize = 5;
    const SENTINEL: u8 = 0xa5;
    let head_size = 256;
    for (page, bs) in [
        (Page::Vllm, 16),
        (Page::Vllm, 32),
        (Page::TokenMajor, 16),
        (Page::TokenMajor, 32),
    ] {
        let Some(pool) = pool(page, head_size, bs, BLOCKS, MetalDtype::Int8) else {
            return;
        };
        let tokens = 3 * bs;
        let slots = slots(tokens, bs, &[4, 1, 2]);
        let keys = rows(tokens, head_size, 0x1234 + bs as u64);
        let values = rows(tokens, head_size, 0x4321 + bs as u64);
        let blank = fill(&pool, BLOCKS, SENTINEL);
        let want = expected(page, &blank, BLOCKS, bs, head_size, &keys, &values, &slots);
        write(&pool, (&keys, &values), &slots, page);
        for (block, want) in want.iter().enumerate() {
            let got = &pool.read_block_all_layers(block as u32).unwrap()[0];
            assert!(got == want, "{page:?} block_size {bs} block {block}");
        }
    }
}

/// The new page's store against the f32 reference (K token-major, V
/// dimension-major, scales at `splash_q8_scale_index`).
#[test]
fn keys_token_major_store_matches_the_reference() {
    const BS: usize = 32;
    const BLOCKS: usize = 4;
    const SENTINEL: u8 = 0x3c;
    let head_size = 256;
    let Some(pool) = pool(
        Page::KeysTokenMajor,
        head_size,
        BS,
        BLOCKS,
        MetalDtype::Int8,
    ) else {
        return;
    };
    let tokens = 2 * BS + 16;
    let slots = slots(tokens, BS, &[3, 0, 1]);
    let keys = rows(tokens, head_size, 0x77);
    let values = rows(tokens, head_size, 0x99);
    let blank = fill(&pool, BLOCKS, SENTINEL);
    let want = expected(
        Page::KeysTokenMajor,
        &blank,
        BLOCKS,
        BS,
        head_size,
        &keys,
        &values,
        &slots,
    );
    write(&pool, (&keys, &values), &slots, Page::KeysTokenMajor);
    for (block, want) in want.iter().enumerate() {
        let got = &pool.read_block_all_layers(block as u32).unwrap()[0];
        assert!(got == want, "block {block}");
    }
}

/// The write validator takes a rank-4 K beside dimension-major values as
/// the new page only for an int8 32-token page of the right shapes.
/// Mutations: the new page's shape check off (token-major V accepted ->
/// red), its int8 / block 32 check off (block 16 or bf16 accepted -> red).
#[test]
fn validator_takes_the_new_page_only_whole() {
    const NB: usize = 3;
    let d = 256;
    let tokens = 3;
    let slots = [0i64, 1, 33];
    let keys = rows(tokens, d, 5);
    let values = rows(tokens, d, 6);
    let rows = (keys.as_slice(), values.as_slice());
    let Some(ktm32) = pool(Page::KeysTokenMajor, d, 32, NB, MetalDtype::Int8) else {
        return;
    };
    let vt32 = pool(Page::TokenMajor, d, 32, NB, MetalDtype::Int8).unwrap();
    let vt16 = pool(Page::TokenMajor, d, 16, NB, MetalDtype::Int8).unwrap();
    let vllm16 = pool(Page::Vllm, d, 16, NB, MetalDtype::Int8).unwrap();
    let vt32_bf16 = pool(Page::TokenMajor, d, 32, NB, MetalDtype::BFloat16).unwrap();
    let vllm32_bf16 = pool(Page::Vllm, d, 32, NB, MetalDtype::BFloat16).unwrap();
    assert!(
        LayerKVPool::new_with_layout(
            vllm16.config().clone(),
            NB as u32,
            NB as u32,
            MetalDtype::Int8,
            Page::KeysTokenMajor.layout(),
        )
        .is_err(),
        "a 16-token keys-token-major pool"
    );
    let (ktm, vt32, vt16, vllm16) = (
        planes(&ktm32),
        planes(&vt32),
        planes(&vt16),
        planes(&vllm16),
    );
    let (vt_bf16, vllm_bf16) = (planes(&vt32_bf16), planes(&vllm32_bf16));

    assert!(
        write_arrays(
            &ktm.key,
            &ktm.value,
            &ktm.key_scales,
            &ktm.value_scales,
            rows,
            &slots,
            32,
            d,
            INT8,
            false
        ),
        "the whole new page"
    );
    assert!(
        !write_arrays(
            &ktm.key,
            &vt32.value,
            &ktm.key_scales,
            &ktm.value_scales,
            rows,
            &slots,
            32,
            d,
            INT8,
            false
        ),
        "rank-4 K beside token-major V"
    );
    assert!(
        !write_arrays(
            &vt16.key,
            &vllm16.value,
            &vllm16.key_scales,
            &vllm16.value_scales,
            rows,
            &[0, 1, 17],
            16,
            d,
            INT8,
            false
        ),
        "rank-4 K at block 16"
    );
    assert!(
        !write_arrays(
            &vt_bf16.key,
            &vllm_bf16.value,
            &vt_bf16.key_scales,
            &vt_bf16.value_scales,
            rows,
            &slots,
            32,
            d,
            BF16,
            false
        ),
        "rank-4 bf16 K beside dimension-major V"
    );
    assert!(
        !write_arrays(
            &ktm.key,
            &ktm.value,
            &ktm.key_scales,
            &ktm.value_scales,
            rows,
            &slots,
            32,
            d,
            INT8,
            true
        ),
        "token-major flag beside dimension-major V"
    );
}
