//! Varlen paged-attention kernel tests.
//!
//! End-to-end verification of the multi-row paged-attention kernel
//! used for MTP speculative decoding through `mlx-sys`'s graph-native C++
//! primitives (`mlx_paged_kv_write_forward`, `mlx_paged_attention_forward`,
//! `mlx_paged_attention_varlen_forward`) backed by `LayerKVPool`.
//!
//! The single-row kernel ships one Q vector per sequence; the varlen kernel
//! ships `cu_seqlens_q[num_seqs]` Q vectors and uses an on-GPU binary search
//! plus per-query causal cutoff
//!
//!     effective_context_len = context_len - q_len + q_pos_in_seq + 1
//!
//! to mask the verify-window draft tokens correctly. Tests:
//!
//! 1. **Parity** — single sequence, T=1. The varlen kernel must produce
//!    byte-equivalent output to the single-row kernel (within bf16
//!    rounding) because q_len=1, q_pos_in_seq=0, effective_context=
//!    context_len.
//! 2. **Ragged batch** — 3 sequences with q_lens [1, 2, 4] and
//!    context_lens [16, 17, 32]. Compared against a host-side reference
//!    that recomputes the causal cutoff per query.
//! 3. **High-partition V2** — context_len = 1024 (exercises V2 reduce
//!    kernel with 2 partitions per query token), ragged Q. Compared
//!    against host reference.
//! 4. **Per-query effective_context** — T=2 with q_len=2 in a single
//!    sequence. The two query tokens see DIFFERENT effective_context
//!    values (one less and exactly context_len), so the outputs MUST
//!    differ. Catches a kernel that ignores cu_seqlens_q.
//!
//! All tests skip cleanly on hosts without Metal.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use mlx_paged_attn::metal::{MetalDtype, MetalState};
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::mlx_array;

const KV_DTYPE_BF16: u8 = 1;

struct Owned(*mut mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

// =============================================================================
// bf16 & MLX C ABI host helpers (mirrors paged_int8_read.rs).
// =============================================================================

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn metal_available() -> bool {
    MetalState::get().is_ok()
}

fn from_bf16(data: &[u16], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_bfloat16(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_f32(data: &[f32], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_float32(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_i32(data: &[i32], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_int32(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_i64(data: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_int64(data.as_ptr(), [data.len() as i64].as_ptr(), 1) })
}

fn to_f32(array: &Owned) -> Vec<f32> {
    let len = unsafe { mlx_sys::mlx_array_size(array.0) };
    let mut out = vec![0.0f32; len];
    assert!(unsafe { mlx_sys::mlx_array_to_float32(array.0, out.as_mut_ptr(), out.len()) });
    out
}

/// A BF16 `LayerKVPool` populated with one or more sequences' synthetic K/V via
/// `mlx_paged_kv_write_forward`.
struct FilledPool {
    _pool: LayerKVPool,
    k_pool: Owned,
    v_pool: Owned,
    k_scale: Owned,
    v_scale: Owned,
    block_table: Vec<i32>,
    max_blocks_per_seq: usize,
}

fn fill_bf16_pool(
    k_per_seq: &[Vec<f32>],
    v_per_seq: &[Vec<f32>],
    context_lens: &[u32],
    head_size: u32,
    num_kv_heads: u32,
    block_size: u32,
) -> FilledPool {
    let max_blocks_per_seq = context_lens
        .iter()
        .map(|&c| c.div_ceil(block_size))
        .max()
        .unwrap();

    let mut block_table = Vec::new();
    let mut seq_start_blocks = Vec::with_capacity(context_lens.len());
    let mut next_block: u32 = 0;
    for &ctx in context_lens {
        let n_blocks = ctx.div_ceil(block_size);
        seq_start_blocks.push(next_block);
        let mut row: Vec<i32> = (next_block..next_block + n_blocks)
            .map(|b| b as i32)
            .collect();
        while row.len() < max_blocks_per_seq as usize {
            row.push((next_block + n_blocks - 1) as i32);
        }
        block_table.extend(row);
        next_block += n_blocks;
    }
    let total_blocks = next_block;

    let config = PagedAttentionConfig {
        block_size,
        gpu_memory_mb: 256,
        head_size,
        num_kv_heads,
        num_layers: 1,
        max_seq_len: Some(max_blocks_per_seq * block_size),
        max_batch_size: Some(context_lens.len() as u32),
    };
    let pool = LayerKVPool::new(config, total_blocks, total_blocks, MetalDtype::BFloat16)
        .expect("LayerKVPool::new must succeed");

    // Concatenate every sequence's K/V and build the matching flat slot_mapping.
    let total_tokens: usize = context_lens.iter().map(|&c| c as usize).sum();
    let stride = (num_kv_heads * head_size) as usize;
    let mut all_k_bf16 = Vec::with_capacity(total_tokens * stride);
    let mut all_v_bf16 = Vec::with_capacity(total_tokens * stride);
    let mut slots = Vec::with_capacity(total_tokens);
    for (s, &ctx) in context_lens.iter().enumerate() {
        let base_block = seq_start_blocks[s] as i64;
        all_k_bf16.extend(k_per_seq[s].iter().copied().map(f32_to_bf16_bits));
        all_v_bf16.extend(v_per_seq[s].iter().copied().map(f32_to_bf16_bits));
        for t in 0..ctx as i64 {
            let block_idx = base_block + t / block_size as i64;
            let block_offset = t % block_size as i64;
            slots.push(block_idx * block_size as i64 + block_offset);
        }
    }

    let kv_shape = [total_tokens as i64, num_kv_heads as i64, head_size as i64];
    let new_k = from_bf16(&all_k_bf16, &kv_shape);
    let new_v = from_bf16(&all_v_bf16, &kv_shape);
    let slot_mapping = from_i64(&slots);
    let k_pool_in = Owned(pool.key_cache_array_raw(0).expect("key_cache_array_raw"));
    let v_pool_in = Owned(
        pool.value_cache_array_raw(0)
            .expect("value_cache_array_raw"),
    );
    let k_scale_in = from_f32(&[1.0], &[1]);
    let v_scale_in = from_f32(&[1.0], &[1]);

    let mut out = [std::ptr::null_mut::<mlx_array>(); 4];
    let [k_out, v_out, ks_out, vs_out] = &mut out;
    let ok = unsafe {
        mlx_sys::mlx_paged_kv_write_forward(
            k_pool_in.0,
            v_pool_in.0,
            new_k.0,
            new_v.0,
            slot_mapping.0,
            k_scale_in.0,
            v_scale_in.0,
            block_size as i32,
            num_kv_heads as i32,
            head_size as i32,
            KV_DTYPE_BF16,
            false,
            k_out,
            v_out,
            ks_out,
            vs_out,
        )
    };
    assert!(ok, "mlx_paged_kv_write_forward must succeed");
    let written = out.map(Owned);
    let mut handles = out;
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 4) });
    let [k_pool, v_pool, k_scale, v_scale] = written;

    FilledPool {
        _pool: pool,
        k_pool,
        v_pool,
        k_scale,
        v_scale,
        block_table,
        max_blocks_per_seq: max_blocks_per_seq as usize,
    }
}

/// Host-side reference for a single query token attending to
/// `[effective_context_len]` KV positions of a single sequence.
fn host_attention_one_token(
    q: &[f32],     // [head_size]
    k_seq: &[f32], // [context_len * head_size]
    v_seq: &[f32], // [context_len * head_size]
    effective_context_len: usize,
    head_size: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; effective_context_len];
    for (t, s) in scores.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for d in 0..head_size {
            acc += q[d] * k_seq[t * head_size + d];
        }
        *s = acc * scale;
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut exps = vec![0.0f32; effective_context_len];
    for (i, e) in exps.iter_mut().enumerate() {
        *e = (scores[i] - max_score).exp();
        sum_exp += *e;
    }
    let inv_sum = 1.0f32 / (sum_exp + 1e-6);
    for e in exps.iter_mut() {
        *e *= inv_sum;
    }
    let mut out = vec![0.0f32; head_size];
    for d in 0..head_size {
        let mut acc = 0.0f32;
        for (t, &w) in exps.iter().enumerate() {
            acc += w * v_seq[t * head_size + d];
        }
        out[d] = acc;
    }
    out
}

/// Deterministic K/V generator for one sequence. Each token has a
/// distinct attention score against the test queries; V values reveal
/// which token contributed.
fn build_synthetic_kv(context_len: usize, head_size: usize, seed: f32) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; context_len * head_size];
    let mut v = vec![0.0f32; context_len * head_size];
    for t in 0..context_len {
        for d in 0..head_size {
            k[t * head_size + d] = ((t as f32) * 0.1 + (d as f32) * 0.01 + seed).sin() * 0.5;
            v[t * head_size + d] = ((t as f32 + 1.0) * 0.1 + (d as f32) * 0.001 + seed * 0.3) * 0.5;
        }
    }
    (k, v)
}

#[allow(clippy::too_many_arguments)]
fn run_varlen(
    filled: &FilledPool,
    q_bf16: &[u16],
    total_queries: usize,
    context_lens: &[u32],
    cu_seqlens_q: &[i32],
    num_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    scale: f32,
) -> Vec<f32> {
    let q_arr = from_bf16(
        q_bf16,
        &[total_queries as i64, num_heads as i64, head_size as i64],
    );
    let block_table = from_i32(
        &filled.block_table,
        &[context_lens.len() as i64, filled.max_blocks_per_seq as i64],
    );
    let seq_lens_i32: Vec<i32> = context_lens.iter().map(|&c| c as i32).collect();
    let seq_lens = from_i32(&seq_lens_i32, &[context_lens.len() as i64]);
    let cu_seqlens = from_i32(cu_seqlens_q, &[cu_seqlens_q.len() as i64]);

    let out = unsafe {
        mlx_sys::mlx_paged_attention_varlen_forward(
            q_arr.0,
            filled.k_pool.0,
            filled.v_pool.0,
            block_table.0,
            seq_lens.0,
            cu_seqlens.0,
            filled.k_scale.0,
            filled.v_scale.0,
            scale,
            /* softcap */ 0.0,
            /* sliding_window */ 0,
            block_size as i32,
            num_heads as i32,
            num_kv_heads as i32,
            head_size as i32,
            KV_DTYPE_BF16,
            std::ptr::null_mut(),
        )
    };
    assert!(
        !out.is_null(),
        "mlx_paged_attention_varlen_forward must succeed"
    );
    let out = Owned(out);
    let mut handles = [out.0];
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 1) });
    to_f32(&out)
}

// =============================================================================
// Test 1 — Parity vs single-row kernel for the single-seq T=1 case.
//
// With q_lens=[1] and cu_seqlens_q=[0, 1]:
//     q_len = 1, q_pos_in_seq = 0 -> effective_context = context_len
// which is exactly the single-row kernel's mask boundary. The two
// kernels must produce equal output (within bf16 rounding noise — both
// take the same QK/V code paths).
// =============================================================================

#[test]
fn varlen_t1_matches_single_row_kernel() {
    if !metal_available() {
        eprintln!("skipping varlen_t1_matches_single_row_kernel: Metal unavailable");
        return;
    }

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;
    const CONTEXT_LEN: u32 = 32;

    let (k_per_token, v_per_token) =
        build_synthetic_kv(CONTEXT_LEN as usize, HEAD_SIZE as usize, 0.0);
    let q: Vec<f32> = (0..HEAD_SIZE).map(|d| 0.7 + 0.01 * d as f32).collect();
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();
    let q_bf16: Vec<u16> = q.iter().copied().map(f32_to_bf16_bits).collect();

    let filled = fill_bf16_pool(
        &[k_per_token],
        &[v_per_token],
        &[CONTEXT_LEN],
        HEAD_SIZE,
        NUM_KV_HEADS,
        BLOCK_SIZE,
    );

    // Single-row baseline.
    let q_arr = from_bf16(&q_bf16, &[1, NUM_HEADS as i64, HEAD_SIZE as i64]);
    let block_table = from_i32(&filled.block_table, &[1, filled.max_blocks_per_seq as i64]);
    let seq_lens = from_i32(&[CONTEXT_LEN as i32], &[1]);
    let baseline_out = unsafe {
        mlx_sys::mlx_paged_attention_forward(
            q_arr.0,
            filled.k_pool.0,
            filled.v_pool.0,
            block_table.0,
            seq_lens.0,
            filled.k_scale.0,
            filled.v_scale.0,
            scale,
            /* softcap */ 0.0,
            /* sliding_window */ 0,
            BLOCK_SIZE as i32,
            NUM_HEADS as i32,
            NUM_KV_HEADS as i32,
            HEAD_SIZE as i32,
            KV_DTYPE_BF16,
        )
    };
    assert!(!baseline_out.is_null(), "single-row dispatch must succeed");
    let baseline_out = Owned(baseline_out);
    let mut handles = [baseline_out.0];
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 1) });
    let baseline_host = to_f32(&baseline_out);

    // Varlen path.
    let varlen_host = run_varlen(
        &filled,
        &q_bf16,
        1,
        &[CONTEXT_LEN],
        &[0, 1],
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        BLOCK_SIZE,
        scale,
    );

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    for d in 0..HEAD_SIZE as usize {
        let diff = (baseline_host[d] - varlen_host[d]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = d;
        }
    }
    let tol = 1.0e-3_f32;
    assert!(
        max_diff < tol,
        "varlen T=1 output diverges from single-row at dim={max_idx} \
         (single_row={}, varlen={}, diff={max_diff} > tol={tol})",
        baseline_host[max_idx],
        varlen_host[max_idx]
    );
}

// =============================================================================
// Test 2 — Ragged batch against host reference.
//
// 3 sequences: q_lens [1, 2, 4], context_lens [16, 17, 32].
// cu_seqlens_q = [0, 1, 3, 7], total_queries = 7.
// =============================================================================

#[test]
fn varlen_ragged_batch_matches_host_reference() {
    if !metal_available() {
        eprintln!("skipping varlen_ragged_batch_matches_host_reference: Metal unavailable");
        return;
    }

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;

    let q_lens = [1u32, 2, 4];
    let context_lens = [16u32, 17, 32];
    let cu_seqlens_q: Vec<i32> = {
        let mut v = vec![0i32];
        let mut acc = 0i32;
        for &q in &q_lens {
            acc += q as i32;
            v.push(acc);
        }
        v
    };
    let total_queries: usize = q_lens.iter().map(|&q| q as usize).sum();

    let mut k_per_seq = Vec::with_capacity(context_lens.len());
    let mut v_per_seq = Vec::with_capacity(context_lens.len());
    for (s, &ctx) in context_lens.iter().enumerate() {
        let (k, v) = build_synthetic_kv(ctx as usize, HEAD_SIZE as usize, s as f32 + 1.0);
        k_per_seq.push(k);
        v_per_seq.push(v);
    }

    let filled = fill_bf16_pool(
        &k_per_seq,
        &v_per_seq,
        &context_lens,
        HEAD_SIZE,
        NUM_KV_HEADS,
        BLOCK_SIZE,
    );

    let mut q_host_f32 = vec![0.0f32; total_queries * HEAD_SIZE as usize];
    for q in 0..total_queries {
        for d in 0..HEAD_SIZE as usize {
            q_host_f32[q * HEAD_SIZE as usize + d] = 0.3 + 0.05 * (q as f32) + 0.005 * (d as f32);
        }
    }
    let q_host_bf16: Vec<u16> = q_host_f32.iter().copied().map(f32_to_bf16_bits).collect();

    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();
    let kernel_out = run_varlen(
        &filled,
        &q_host_bf16,
        total_queries,
        &context_lens,
        &cu_seqlens_q,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        BLOCK_SIZE,
        scale,
    );

    let tol = 5.0e-2_f32;
    for q_token_idx in 0..total_queries {
        let seq_idx = cu_seqlens_q
            .iter()
            .position(|&v| v > q_token_idx as i32)
            .map(|i| i - 1)
            .expect("q_token_idx out of range");
        let q_seq_start = cu_seqlens_q[seq_idx] as usize;
        let q_len = (cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]) as usize;
        let q_pos_in_seq = q_token_idx - q_seq_start;
        let context_len = context_lens[seq_idx] as usize;
        let effective_ctx = context_len - q_len + q_pos_in_seq + 1;

        let q_slice =
            &q_host_f32[q_token_idx * HEAD_SIZE as usize..(q_token_idx + 1) * HEAD_SIZE as usize];
        let q_round: Vec<f32> = q_slice
            .iter()
            .map(|&x| bf16_bits_to_f32(f32_to_bf16_bits(x)))
            .collect();

        let host_ref = host_attention_one_token(
            &q_round,
            &k_per_seq[seq_idx],
            &v_per_seq[seq_idx],
            effective_ctx,
            HEAD_SIZE as usize,
            scale,
        );

        let kernel_slice =
            &kernel_out[q_token_idx * HEAD_SIZE as usize..(q_token_idx + 1) * HEAD_SIZE as usize];

        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        for d in 0..HEAD_SIZE as usize {
            let diff = (kernel_slice[d] - host_ref[d]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = d;
            }
        }
        assert!(
            max_diff < tol,
            "ragged q_token_idx={q_token_idx} (seq={seq_idx}, q_pos={q_pos_in_seq}, \
             effective_ctx={effective_ctx}) diverges at dim={max_idx} \
             (kernel={}, ref={}, diff={max_diff} > tol={tol})",
            kernel_slice[max_idx],
            host_ref[max_idx]
        );
    }
}

// =============================================================================
// Test 3 — High-partition V2 path with ragged Q.
//
// context_len = 1024 (well past PARTITION_SIZE=512, so the V2 reduce
// kernel runs with 2 partitions per query token). 2 sequences with
// q_lens [1, 3] — enough to exercise per-token effective_context.
// =============================================================================

#[test]
fn varlen_v2_high_partition_matches_host_reference() {
    if !metal_available() {
        eprintln!("skipping varlen_v2_high_partition_matches_host_reference: Metal unavailable");
        return;
    }

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;

    let q_lens = [1u32, 3];
    let context_lens = [1024u32, 1024];
    let cu_seqlens_q = [0i32, 1, 4];
    let total_queries: usize = q_lens.iter().map(|&q| q as usize).sum();

    let mut k_per_seq = Vec::with_capacity(context_lens.len());
    let mut v_per_seq = Vec::with_capacity(context_lens.len());
    for (s, &ctx) in context_lens.iter().enumerate() {
        let (k, v) = build_synthetic_kv(ctx as usize, HEAD_SIZE as usize, s as f32 + 2.0);
        k_per_seq.push(k);
        v_per_seq.push(v);
    }

    let filled = fill_bf16_pool(
        &k_per_seq,
        &v_per_seq,
        &context_lens,
        HEAD_SIZE,
        NUM_KV_HEADS,
        BLOCK_SIZE,
    );

    let mut q_host_f32 = vec![0.0f32; total_queries * HEAD_SIZE as usize];
    for q in 0..total_queries {
        for d in 0..HEAD_SIZE as usize {
            q_host_f32[q * HEAD_SIZE as usize + d] = 0.2 + 0.07 * (q as f32) + 0.003 * (d as f32);
        }
    }
    let q_host_bf16: Vec<u16> = q_host_f32.iter().copied().map(f32_to_bf16_bits).collect();

    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();
    let kernel_out = run_varlen(
        &filled,
        &q_host_bf16,
        total_queries,
        &context_lens,
        &cu_seqlens_q,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        BLOCK_SIZE,
        scale,
    );

    let rel_tol = 1.5e-2_f32;
    let abs_tol = 5.0e-2_f32;
    for q_token_idx in 0..total_queries {
        let seq_idx = cu_seqlens_q
            .iter()
            .position(|&v| v > q_token_idx as i32)
            .map(|i| i - 1)
            .expect("q_token_idx out of range");
        let q_seq_start = cu_seqlens_q[seq_idx] as usize;
        let q_len = (cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]) as usize;
        let q_pos_in_seq = q_token_idx - q_seq_start;
        let context_len = context_lens[seq_idx] as usize;
        let effective_ctx = context_len - q_len + q_pos_in_seq + 1;

        let q_slice =
            &q_host_f32[q_token_idx * HEAD_SIZE as usize..(q_token_idx + 1) * HEAD_SIZE as usize];
        let q_round: Vec<f32> = q_slice
            .iter()
            .map(|&x| bf16_bits_to_f32(f32_to_bf16_bits(x)))
            .collect();
        let host_ref = host_attention_one_token(
            &q_round,
            &k_per_seq[seq_idx],
            &v_per_seq[seq_idx],
            effective_ctx,
            HEAD_SIZE as usize,
            scale,
        );

        let kernel_slice =
            &kernel_out[q_token_idx * HEAD_SIZE as usize..(q_token_idx + 1) * HEAD_SIZE as usize];

        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        let mut max_diff_tol = 0.0f32;
        for d in 0..HEAD_SIZE as usize {
            let diff = (kernel_slice[d] - host_ref[d]).abs();
            let elem_tol = abs_tol.max(rel_tol * host_ref[d].abs());
            if diff > max_diff {
                max_diff = diff;
                max_idx = d;
                max_diff_tol = elem_tol;
            }
        }
        assert!(
            max_diff < max_diff_tol,
            "v2 ragged q_token_idx={q_token_idx} (seq={seq_idx}, eff_ctx={effective_ctx}) \
             diverges at dim={max_idx} (kernel={}, ref={}, diff={max_diff} > \
             tol={max_diff_tol})",
            kernel_slice[max_idx],
            host_ref[max_idx]
        );
    }
}

// =============================================================================
// Test 4 — Per-query causal cutoff.
//
// Single sequence with q_len=2, context_len=32. cu_seqlens_q=[0, 2],
// total_queries=2. The two query tokens have DIFFERENT effective_ctx:
//   q=0 (q_pos=0): eff = 32 - 2 + 0 + 1 = 31
//   q=1 (q_pos=1): eff = 32 - 2 + 1 + 1 = 32
// =============================================================================

#[test]
fn varlen_per_query_causal_cutoff_is_enforced() {
    if !metal_available() {
        eprintln!("skipping varlen_per_query_causal_cutoff_is_enforced: Metal unavailable");
        return;
    }

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;
    const CONTEXT_LEN: u32 = 32;

    let (k_per_token, mut v_per_token) =
        build_synthetic_kv(CONTEXT_LEN as usize, HEAD_SIZE as usize, 3.0);
    // Make KV[31] very large in V so a kernel that incorrectly
    // includes it in q=0's attention would shift the output noticeably.
    for d in 0..HEAD_SIZE as usize {
        v_per_token[31 * HEAD_SIZE as usize + d] = 100.0 + d as f32;
    }

    let mut q_host_f32 = vec![0.0f32; 2 * HEAD_SIZE as usize];
    for d in 0..HEAD_SIZE as usize {
        q_host_f32[d] = 0.5 + 0.01 * d as f32;
        q_host_f32[HEAD_SIZE as usize + d] = -0.5 + 0.02 * d as f32;
    }
    let q_bf16: Vec<u16> = q_host_f32.iter().copied().map(f32_to_bf16_bits).collect();

    let filled = fill_bf16_pool(
        &[k_per_token.clone()],
        &[v_per_token.clone()],
        &[CONTEXT_LEN],
        HEAD_SIZE,
        NUM_KV_HEADS,
        BLOCK_SIZE,
    );

    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();
    let kernel_out = run_varlen(
        &filled,
        &q_bf16,
        2,
        &[CONTEXT_LEN],
        &[0, 2],
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        BLOCK_SIZE,
        scale,
    );
    let q0_out = &kernel_out[0..HEAD_SIZE as usize];
    let q1_out = &kernel_out[HEAD_SIZE as usize..2 * HEAD_SIZE as usize];

    let q01_diff = q0_out
        .iter()
        .zip(q1_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        q01_diff > 1.0e-2,
        "q0 and q1 outputs are nearly identical (max_diff={q01_diff}); the kernel \
         is likely ignoring per-query effective_context_len"
    );

    let q_round: Vec<f32> = q_host_f32[..HEAD_SIZE as usize]
        .iter()
        .map(|&x| bf16_bits_to_f32(f32_to_bf16_bits(x)))
        .collect();
    let q0_ref = host_attention_one_token(
        &q_round,
        &k_per_token,
        &v_per_token,
        31,
        HEAD_SIZE as usize,
        scale,
    );
    let mut q0_max_diff = 0.0f32;
    for d in 0..HEAD_SIZE as usize {
        let diff = (q0_out[d] - q0_ref[d]).abs();
        if diff > q0_max_diff {
            q0_max_diff = diff;
        }
    }
    let tol = 5.0e-2_f32;
    assert!(
        q0_max_diff < tol,
        "q=0 output disagrees with host reference that excludes KV[31] \
         (max_diff={q0_max_diff} > tol={tol}); the kernel is leaking KV[31] \
         into the early-window query"
    );

    let q0_max_magnitude = q0_out
        .iter()
        .copied()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(
        q0_max_magnitude < 5.0,
        "q=0 output magnitude {q0_max_magnitude} suggests leakage from KV[31] \
         (V[31] = 100+, softmax weight ~0.03 would push output to ~3+)"
    );
}
