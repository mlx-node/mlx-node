//! Int8 paged store (`PagedKVWrite` with kv_dtype Int8) through the C ABI
//! `mlx_paged_kv_write_forward`, against the flat quantizer
//! `mlx_kv_int8_quantize_rows` and against the pool's index formulas
//! (Splash `splash_store_kv_row`, `splash_kv_value_element`,
//! `splash_q8_scale_index`; K keeps the vLLM x = 16 packing).

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::mlx_array;

const H: usize = 4;
const BS: usize = 16;
const X: usize = 16;

struct Owned(*mut mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

fn config(head_size: usize, num_blocks: usize) -> PagedAttentionConfig {
    PagedAttentionConfig {
        block_size: BS as u32,
        gpu_memory_mb: 256,
        head_size: head_size as u32,
        num_kv_heads: H as u32,
        num_layers: 1,
        max_seq_len: Some((num_blocks * BS) as u32),
        max_batch_size: Some(1),
    }
}

fn pool(head_size: usize, num_blocks: usize, dtype: MetalDtype) -> Option<LayerKVPool> {
    match LayerKVPool::new(
        config(head_size, num_blocks),
        num_blocks as u32,
        num_blocks as u32,
        dtype,
    ) {
        Ok(pool) => Some(pool),
        Err(e) if e.contains("No Metal device found") => None,
        Err(e) => panic!("LayerKVPool::new: {e}"),
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

fn size(array: &Owned) -> usize {
    unsafe { mlx_sys::mlx_array_size(array.0) }
}

fn bytes(array: &Owned, signed: bool) -> Vec<u8> {
    let mut out = vec![0u8; size(array)];
    let ok = unsafe {
        if signed {
            mlx_sys::mlx_array_to_int8(array.0, out.as_mut_ptr() as *mut i8, out.len())
        } else {
            mlx_sys::mlx_array_to_uint8(array.0, out.as_mut_ptr(), out.len())
        }
    };
    assert!(ok, "read back bytes");
    out
}

fn floats(array: &Owned) -> Vec<f32> {
    let mut out = vec![0f32; size(array)];
    assert!(unsafe { mlx_sys::mlx_array_to_float32(array.0, out.as_mut_ptr(), out.len()) });
    out
}

/// Pool K element of (block, head, page token, dimension): vLLM
/// `[nb, H, D/16, bs, 16]`.
fn key_index(block: usize, head: usize, token: usize, d: usize, head_size: usize) -> usize {
    (((block * H + head) * (head_size / X) + d / X) * BS + token) * X + d % X
}

/// Splash `splash_kv_value_element` with page = block: `[nb, H, D, bs]`.
fn value_index(block: usize, head: usize, token: usize, d: usize, head_size: usize) -> usize {
    ((block * H + head) * head_size + d) * BS + token
}

/// Splash `splash_q8_scale_index` with page = block: `[nb, H, bs]`.
fn scale_index(block: usize, head: usize, token: usize) -> usize {
    (block * H + head) * BS + token
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

/// Every output of one `mlx_paged_kv_write_forward`, evaluated.
struct Written {
    key: Owned,
    value: Owned,
    key_scales: Owned,
    value_scales: Owned,
}

#[allow(clippy::too_many_arguments)]
fn write(
    pool: &LayerKVPool,
    keys: &[u16],
    values: &[u16],
    slots: &[i64],
    head_size: usize,
    kv_dtype: u8,
) -> Result<Written, String> {
    let tokens = slots.len() as i64;
    let shape = [tokens, H as i64, head_size as i64];
    let new_k = from_bf16(keys, &shape);
    let new_v = from_bf16(values, &shape);
    let slots = from_i64(slots);
    let key = Owned(pool.key_cache_array_raw(0)?);
    let value = Owned(pool.value_cache_array_raw(0)?);
    let (key_scale, value_scale) = if kv_dtype == 3 {
        (
            Owned(pool.key_scale_array_raw(0)?),
            Owned(pool.value_scale_array_raw(0)?),
        )
    } else {
        (from_f32(&[1.0], &[1]), from_f32(&[1.0], &[1]))
    };
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
            head_size as i32,
            kv_dtype,
            k_out,
            v_out,
            ks_out,
            vs_out,
        )
    };
    let written = Written {
        key: Owned(out[0]),
        value: Owned(out[1]),
        key_scales: Owned(out[2]),
        value_scales: Owned(out[3]),
    };
    if !ok {
        return Err("mlx_paged_kv_write_forward refused".into());
    }
    let mut handles = out;
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 4) });
    Ok(written)
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

/// `[T, H, 256]` bf16 rows of every kind the parity gate needs: seeded
/// random rows at several magnitudes, an all-zero row, rows with max 127
/// holding the exact ties ±0.5, ±1.5, ±2.5, and rows whose largest
/// magnitude is negative.
fn parity_rows(tokens: usize, seed: u64) -> Vec<u16> {
    const D: usize = 256;
    let mut rng = Rng(seed);
    let mut out = vec![0u16; tokens * H * D];
    for t in 0..tokens {
        for h in 0..H {
            let row = &mut out[(t * H + h) * D..(t * H + h + 1) * D];
            match (t * H + h) % 7 {
                0 if t == 3 => {}
                1 => {
                    for (d, x) in row.iter_mut().enumerate() {
                        let tie = [0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5][d % 6];
                        *x = bf16(if d == 0 { 127.0 } else { tie });
                    }
                }
                2 => {
                    for (d, x) in row.iter_mut().enumerate() {
                        *x = bf16(if d == 17 { -9.75 } else { rng.next() * 9.0 });
                    }
                }
                kind => {
                    let magnitude = [0.01f32, 1.0, 3.7, 40.0, 1e-3, 250.0, 0.3][kind];
                    for x in row.iter_mut() {
                        *x = bf16(rng.next() * magnitude);
                    }
                }
            }
        }
    }
    out
}

/// `[T, H, D]` -> `[1, H, T, D]` for the flat quantizer.
fn head_major(rows: &[u16], tokens: usize, head_size: usize) -> Vec<u16> {
    let mut out = vec![0u16; rows.len()];
    for t in 0..tokens {
        for h in 0..H {
            let src = (t * H + h) * head_size;
            let dst = (h * tokens + t) * head_size;
            out[dst..dst + head_size].copy_from_slice(&rows[src..src + head_size]);
        }
    }
    out
}

fn flat_quantize(rows: &[u16], tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let x = from_bf16(
        &head_major(rows, tokens, 256),
        &[1, H as i64, tokens as i64, 256],
    );
    let mut q = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    assert_eq!(
        unsafe { mlx_sys::mlx_kv_int8_quantize_rows(x.0, &mut q, &mut s) },
        0
    );
    let (q, s) = (Owned(q), Owned(s));
    (bytes(&q, true), floats(&s))
}

/// Parity gate: the int8 paged store writes the flat quantizer's bytes and
/// scale bits for the same bf16 rows.
///
/// Mutations K1 (`rint` -> `trunc`) and K2 (127 -> 127.5) go red. A
/// `-ffast-math` build of the store (Splash's build mode) differs in 20 of
/// 196,608 q bytes here (exact .5 quotients), none of the scales.
#[test]
fn paged_int8_store_matches_the_flat_quantizer() {
    const D: usize = 256;
    const BLOCKS: usize = 8;
    let Some(pool) = pool(D, BLOCKS, MetalDtype::Int8) else {
        return;
    };
    let tokens = 96;
    // Every slot of blocks 1..=6, out of order: block-reversed, token-strided.
    let slots: Vec<i64> = (0..tokens)
        .map(|t| {
            let block = 6 - t / BS;
            let token = (t * 5) % BS;
            (block * BS + token) as i64
        })
        .collect();
    let keys = parity_rows(tokens, 0x5eed);
    let values = parity_rows(tokens, 0xfeed);
    let written = write(&pool, &keys, &values, &slots, D, 3).unwrap();
    let pool_key = bytes(&written.key, true);
    let pool_value = bytes(&written.value, true);
    let pool_key_scales = floats(&written.key_scales);
    let pool_value_scales = floats(&written.value_scales);
    let (ref_key, ref_key_scales) = flat_quantize(&keys, tokens);
    let (ref_value, ref_value_scales) = flat_quantize(&values, tokens);

    let mut q_mismatch = 0usize;
    let mut scale_mismatch = 0usize;
    let mut rows_hit = 0usize;
    for (t, &slot) in slots.iter().enumerate() {
        let (block, token) = (slot as usize / BS, slot as usize % BS);
        for h in 0..H {
            let row = h * tokens + t;
            let mut row_differs = false;
            for d in 0..D {
                let k = pool_key[key_index(block, h, token, d, D)];
                let v = pool_value[value_index(block, h, token, d, D)];
                if k != ref_key[row * D + d] {
                    if q_mismatch < 5 {
                        println!(
                            "K t{t} h{h} d{d}: bf16 {} store {} flat {} scale {}",
                            f32_of(keys[(t * H + h) * D + d]),
                            k as i8,
                            ref_key[row * D + d] as i8,
                            ref_key_scales[row]
                        );
                    }
                    q_mismatch += 1;
                    row_differs = true;
                }
                if v != ref_value[row * D + d] {
                    q_mismatch += 1;
                    row_differs = true;
                }
            }
            let at = scale_index(block, h, token);
            if pool_key_scales[at].to_bits() != ref_key_scales[row].to_bits() {
                scale_mismatch += 1;
                row_differs = true;
            }
            if pool_value_scales[at].to_bits() != ref_value_scales[row].to_bits() {
                scale_mismatch += 1;
                row_differs = true;
            }
            rows_hit += usize::from(row_differs);
        }
    }
    println!(
        "int8 store vs flat quantizer: {q_mismatch} of {} q bytes, {scale_mismatch} of {} scales, \
         {rows_hit} of {} rows differ",
        2 * tokens * H * D,
        2 * tokens * H,
        2 * tokens * H
    );
    assert_eq!((q_mismatch, scale_mismatch), (0, 0));
    // The rows exercise what they claim: (token 0, head 1) holds the exact
    // ties (half to even), (token 3, head 2) is all zero.
    let tie_row = tokens;
    let tie_q: Vec<i8> = (1..7).map(|d| ref_key[tie_row * D + d] as i8).collect();
    assert_eq!(tie_q, [2, 2, 0, -2, -2, 0]);
    let zero_row = 2 * tokens + 3;
    assert_eq!(ref_key_scales[zero_row].to_bits(), 0);
    assert!(
        ref_key[zero_row * D..(zero_row + 1) * D]
            .iter()
            .all(|&q| q == 0)
    );
}

/// `[T, H, D]` integer-valued bf16 rows in -15..=15 (exact in bf16), with
/// the row (token 2, head 1) all zero.
fn layout_rows(tokens: usize, head_size: usize, side: usize) -> Vec<u16> {
    let mut out = vec![0u16; tokens * H * head_size];
    for t in 0..tokens {
        for h in 0..H {
            if (t, h) == (2, 1) {
                continue;
            }
            for d in 0..head_size {
                let v = ((t * 7 + h * 5 + d * 3 + side * 11) % 31) as f32 - 15.0;
                out[(t * H + h) * head_size + d] = bf16(v);
            }
        }
    }
    out
}

fn fill(pool: &LayerKVPool, blocks: usize, byte: u8) -> (Vec<u8>, Vec<u8>) {
    let (k, v) = pool.block_blob_bytes().unwrap();
    let (k, v) = (vec![byte; k as usize], vec![byte; v as usize]);
    let ids: Vec<u32> = (0..blocks as u32).collect();
    pool.write_blocks_from_host(0, &ids, &k.repeat(blocks), &v.repeat(blocks))
        .unwrap();
    (k, v)
}

/// Layout gate, kernel half: every byte of K, V and both scale planes of
/// every block equals the f32 reference at the index formulas; untouched
/// slots (including the token with slot -1) keep the host-IO sentinel; the
/// MLX output views read the same bytes.
///
/// Mutation K3 (the kernel writes V token-major) goes red.
#[test]
fn paged_int8_store_layout_matches_the_formulas() {
    const BLOCKS: usize = 6;
    const SENTINEL: u8 = 0x5a;
    for head_size in [128usize, 256] {
        let Some(int8) = pool(head_size, BLOCKS, MetalDtype::Int8) else {
            return;
        };
        let tokens = 40;
        // Blocks 5, 0, 2 in turn; token 7 is skipped (slot -1).
        let slots: Vec<i64> = (0..tokens)
            .map(|t| {
                if t == 7 {
                    -1
                } else {
                    ([5usize, 0, 2][t % 3] * BS + t / 3) as i64
                }
            })
            .collect();
        let keys = layout_rows(tokens, head_size, 0);
        let values = layout_rows(tokens, head_size, 1);

        let (k_blob, v_blob) = fill(&int8, BLOCKS, SENTINEL);
        let data = H * head_size * BS;
        assert_eq!(
            (k_blob.len(), v_blob.len()),
            (data + H * BS * 4, data + H * BS * 4)
        );
        let mut expected: Vec<(Vec<u8>, Vec<u8>)> = vec![(k_blob, v_blob); BLOCKS];
        for (t, &slot) in slots.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let (block, token) = (slot as usize / BS, slot as usize % BS);
            for h in 0..H {
                let start = (t * H + h) * head_size;
                for (side, rows) in [&keys, &values].into_iter().enumerate() {
                    let row: Vec<f32> = rows[start..start + head_size]
                        .iter()
                        .map(|&b| f32_of(b))
                        .collect();
                    let (q, scale) = quantize(&row);
                    let blob = if side == 0 {
                        &mut expected[block].0
                    } else {
                        &mut expected[block].1
                    };
                    for (d, &q) in q.iter().enumerate() {
                        let at = if side == 0 {
                            key_index(0, h, token, d, head_size)
                        } else {
                            value_index(0, h, token, d, head_size)
                        };
                        blob[at] = q as u8;
                    }
                    let at = data + scale_index(0, h, token) * 4;
                    blob[at..at + 4].copy_from_slice(&scale.to_ne_bytes());
                }
            }
        }

        let written = write(&int8, &keys, &values, &slots, head_size, 3).unwrap();
        for (block, want) in expected.iter().enumerate() {
            let got = &int8.read_block_all_layers(block as u32).unwrap()[0];
            assert!(got.0 == want.0, "D {head_size} block {block}: K blob");
            assert!(got.1 == want.1, "D {head_size} block {block}: V blob");
        }
        let view_bits =
            |floats: Vec<f32>| -> Vec<u8> { floats.iter().flat_map(|f| f.to_ne_bytes()).collect() };
        let views = [
            bytes(&written.key, true),
            bytes(&written.value, true),
            view_bits(floats(&written.key_scales)),
            view_bits(floats(&written.value_scales)),
        ];
        for (block, want) in expected.iter().enumerate() {
            let scales = H * BS * 4;
            assert_eq!(&views[0][block * data..(block + 1) * data], &want.0[..data]);
            assert_eq!(&views[1][block * data..(block + 1) * data], &want.1[..data]);
            assert_eq!(
                &views[2][block * scales..(block + 1) * scales],
                &want.0[data..]
            );
            assert_eq!(
                &views[3][block * scales..(block + 1) * scales],
                &want.1[data..]
            );
        }
    }
}
