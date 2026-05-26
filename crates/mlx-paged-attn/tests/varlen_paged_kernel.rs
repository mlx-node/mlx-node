//! Phase 4a varlen paged-attention kernel tests.
//!
//! End-to-end verification of the multi-row paged-attention kernel
//! introduced for MTP speculative decoding. The single-row kernel ships
//! one Q vector per sequence; the varlen kernel ships
//! `cu_seqlens_q[num_seqs]` Q vectors and uses an on-GPU binary search
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

#![cfg(target_os = "macos")]

use metal::MTLResourceOptions;
use metal::foreign_types::ForeignType;
use mlx_paged_attn::metal::{
    MetalDtype, MetalState, PagedAttentionParams, PagedAttentionVarlenParams, RawBufferInfo,
    dispatch_paged_attention_v1_raw, dispatch_paged_attention_varlen_auto,
    dispatch_paged_attention_varlen_v1_raw,
};

// =============================================================================
// bf16 host helpers (mirrors sliding_window_kernel.rs).
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

// =============================================================================
// K/V pool builders. Layouts mirror the single-row kernel exactly because
// the varlen kernel reuses the same K/V pool format — only the Q layout
// changes.
// =============================================================================

/// `[num_blocks, num_kv_heads, head_size/x_pack, block_size, x_pack]`
fn build_k_pool_bf16(
    k_per_token: &[f32],
    context_len: usize,
    head_size: usize,
    num_kv_heads: usize,
    block_size: usize,
    num_blocks: usize,
    x_pack: usize,
) -> Vec<u16> {
    let head_per_block = (head_size / x_pack) * block_size * x_pack;
    let stride_block = num_kv_heads * head_per_block;
    let stride_head = head_per_block;
    let stride_xidx = block_size * x_pack;
    let stride_blockoff = x_pack;
    let mut pool = vec![0u16; num_blocks * stride_block];
    for t in 0..context_len {
        let block_idx = t / block_size;
        let block_offset = t % block_size;
        for h in 0..num_kv_heads {
            for j in 0..head_size {
                let x_idx = j / x_pack;
                let x_offset = j % x_pack;
                let pool_idx = block_idx * stride_block
                    + h * stride_head
                    + x_idx * stride_xidx
                    + block_offset * stride_blockoff
                    + x_offset;
                let token_idx = t * num_kv_heads * head_size + h * head_size + j;
                pool[pool_idx] = f32_to_bf16_bits(k_per_token[token_idx]);
            }
        }
    }
    pool
}

/// `[num_blocks, num_kv_heads, head_size, block_size]`
fn build_v_pool_bf16(
    v_per_token: &[f32],
    context_len: usize,
    head_size: usize,
    num_kv_heads: usize,
    block_size: usize,
    num_blocks: usize,
) -> Vec<u16> {
    let head_per_block = head_size * block_size;
    let stride_block = num_kv_heads * head_per_block;
    let stride_head = head_per_block;
    let stride_j = block_size;
    let mut pool = vec![0u16; num_blocks * stride_block];
    for t in 0..context_len {
        let block_idx = t / block_size;
        let block_offset = t % block_size;
        for h in 0..num_kv_heads {
            for j in 0..head_size {
                let pool_idx =
                    block_idx * stride_block + h * stride_head + j * stride_j + block_offset;
                let token_idx = t * num_kv_heads * head_size + h * head_size + j;
                pool[pool_idx] = f32_to_bf16_bits(v_per_token[token_idx]);
            }
        }
    }
    pool
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
    let state = MetalState::get().expect("Metal must be available");

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;
    const CONTEXT_LEN: u32 = 32;
    const NUM_BLOCKS: u32 = 2;
    const X_PACK: u32 = 8;

    let (k_per_token, v_per_token) =
        build_synthetic_kv(CONTEXT_LEN as usize, HEAD_SIZE as usize, 0.0);
    let q: Vec<f32> = (0..HEAD_SIZE).map(|d| 0.7 + 0.01 * d as f32).collect();
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();

    let q_bf16: Vec<u16> = q.iter().copied().map(f32_to_bf16_bits).collect();
    let k_pool = build_k_pool_bf16(
        &k_per_token,
        CONTEXT_LEN as usize,
        HEAD_SIZE as usize,
        NUM_KV_HEADS as usize,
        BLOCK_SIZE as usize,
        NUM_BLOCKS as usize,
        X_PACK as usize,
    );
    let v_pool = build_v_pool_bf16(
        &v_per_token,
        CONTEXT_LEN as usize,
        HEAD_SIZE as usize,
        NUM_KV_HEADS as usize,
        BLOCK_SIZE as usize,
        NUM_BLOCKS as usize,
    );

    let block_table: Vec<u32> = vec![0, 1];
    let seq_lens: Vec<u32> = vec![CONTEXT_LEN];
    let cu_seqlens_q: Vec<i32> = vec![0, 1];

    let key_pool = state.device.new_buffer_with_data(
        k_pool.as_ptr() as *const _,
        std::mem::size_of_val(k_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let value_pool = state.device.new_buffer_with_data(
        v_pool.as_ptr() as *const _,
        std::mem::size_of_val(v_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let q_buf = state.device.new_buffer_with_data(
        q_bf16.as_ptr() as *const _,
        std::mem::size_of_val(q_bf16.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_table_buf = state.device.new_buffer_with_data(
        block_table.as_ptr() as *const _,
        std::mem::size_of_val(block_table.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_lens_buf = state.device.new_buffer_with_data(
        seq_lens.as_ptr() as *const _,
        std::mem::size_of_val(seq_lens.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cu_seqlens_q_buf = state.device.new_buffer_with_data(
        cu_seqlens_q.as_ptr() as *const _,
        std::mem::size_of_val(cu_seqlens_q.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let q_stride = (NUM_HEADS * HEAD_SIZE) as i32;
    let kv_block_stride = (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE) as i32;
    let kv_head_stride = (HEAD_SIZE * BLOCK_SIZE) as i32;

    // Single-row baseline.
    let baseline_params = PagedAttentionParams {
        num_seqs: 1,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: CONTEXT_LEN,
        max_num_blocks_per_seq: block_table.len() as u32,
        scale,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let q_raw = RawBufferInfo {
        ptr: q_buf.as_ptr() as *mut std::ffi::c_void,
        offset: 0,
    };
    let baseline_out = unsafe {
        dispatch_paged_attention_v1_raw(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table_buf,
            &seq_lens_buf,
            &baseline_params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("single-row dispatch must succeed");

    // Varlen path.
    let varlen_params = PagedAttentionVarlenParams {
        num_seqs: 1,
        total_queries: 1,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: CONTEXT_LEN,
        max_num_blocks_per_seq: block_table.len() as u32,
        scale,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let varlen_out = unsafe {
        dispatch_paged_attention_varlen_v1_raw(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table_buf,
            &seq_lens_buf,
            &cu_seqlens_q_buf,
            &varlen_params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("varlen dispatch must succeed");

    let baseline_host = read_output_bf16(state, &baseline_out.buffer, HEAD_SIZE as usize);
    let varlen_host = read_output_bf16(state, &varlen_out.buffer, HEAD_SIZE as usize);

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    for d in 0..HEAD_SIZE as usize {
        let diff = (baseline_host[d] - varlen_host[d]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = d;
        }
    }
    // Both kernels perform fp32 accumulation over bf16 inputs and write
    // bf16 output. The only divergence is potentially different reduction
    // ordering inside the warps; in practice with identical NUM_THREADS
    // and NUM_WARPS the order is identical and the bit pattern matches.
    // Allow a generous safety margin for any future tuning differences.
    let tol = 1.0e-3_f32;
    assert!(
        max_diff < tol,
        "varlen T=1 output diverges from single-row at dim={max_idx} \
         (single_row={}, varlen={}, diff={max_diff} > tol={tol})",
        baseline_host[max_idx],
        varlen_host[max_idx]
    );
}

fn read_output_bf16(state: &MetalState, buf: &metal::Buffer, count: usize) -> Vec<f32> {
    let bytes = count * std::mem::size_of::<u16>();
    let shared = state
        .device
        .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let cmd = state.command_queue.new_command_buffer();
    let blit = cmd.new_blit_command_encoder();
    blit.copy_from_buffer(buf, 0, &shared, 0, bytes as u64);
    blit.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let slice = unsafe { std::slice::from_raw_parts(shared.contents() as *const u16, count) };
    slice.iter().map(|&b| bf16_bits_to_f32(b)).collect()
}

// =============================================================================
// Test 2 — Ragged batch against host reference.
//
// 3 sequences: q_lens [1, 2, 4], context_lens [16, 17, 32].
// cu_seqlens_q = [0, 1, 3, 7], total_queries = 7.
//
// For each query token i we compute:
//     seq_idx = find_seq_idx(cu_seqlens_q, i)
//     q_pos_in_seq = i - cu_seqlens_q[seq_idx]
//     q_len = cu_seqlens_q[seq_idx+1] - cu_seqlens_q[seq_idx]
//     effective_ctx = context_lens[seq_idx] - q_len + q_pos_in_seq + 1
// and reference-compute attention restricted to the first effective_ctx
// KV positions.
//
// Expected effective_ctx values:
//   q=0: seq=0, q_pos=0, q_len=1, ctx=16 -> eff = 16
//   q=1: seq=1, q_pos=0, q_len=2, ctx=17 -> eff = 16
//   q=2: seq=1, q_pos=1, q_len=2, ctx=17 -> eff = 17
//   q=3: seq=2, q_pos=0, q_len=4, ctx=32 -> eff = 29
//   q=4: seq=2, q_pos=1, q_len=4, ctx=32 -> eff = 30
//   q=5: seq=2, q_pos=2, q_len=4, ctx=32 -> eff = 31
//   q=6: seq=2, q_pos=3, q_len=4, ctx=32 -> eff = 32
// =============================================================================

#[test]
fn varlen_ragged_batch_matches_host_reference() {
    if !metal_available() {
        eprintln!("skipping varlen_ragged_batch_matches_host_reference: Metal unavailable");
        return;
    }
    let state = MetalState::get().expect("Metal must be available");

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;
    const X_PACK: u32 = 8;

    let q_lens = vec![1u32, 2, 4];
    let context_lens = vec![16u32, 17, 32];
    let cu_seqlens_q: Vec<i32> = {
        let mut v = vec![0i32];
        let mut acc = 0i32;
        for &q in &q_lens {
            acc += q as i32;
            v.push(acc);
        }
        v
    };
    let total_queries: u32 = q_lens.iter().sum();
    let num_seqs: u32 = q_lens.len() as u32;
    let max_context_len: u32 = *context_lens.iter().max().unwrap();

    // Per-sequence K/V (each gets its own seed so attention scores
    // depend on which sequence the kernel ends up looking up).
    let mut k_per_seq: Vec<Vec<f32>> = Vec::with_capacity(num_seqs as usize);
    let mut v_per_seq: Vec<Vec<f32>> = Vec::with_capacity(num_seqs as usize);
    for (s, &ctx) in context_lens.iter().enumerate() {
        let (k, v) = build_synthetic_kv(ctx as usize, HEAD_SIZE as usize, s as f32 + 1.0);
        k_per_seq.push(k);
        v_per_seq.push(v);
    }

    // Lay out a SHARED pool with enough blocks for all sequences. Each
    // sequence gets ceil(ctx / BLOCK_SIZE) blocks. Block-table maps
    // global block indices.
    let mut block_table: Vec<u32> = Vec::new();
    let mut blocks_per_seq: Vec<u32> = Vec::new();
    let max_blocks_per_seq: u32 = context_lens
        .iter()
        .map(|&c| c.div_ceil(BLOCK_SIZE))
        .max()
        .unwrap();
    let mut next_block: u32 = 0;
    for &ctx in &context_lens {
        let n_blocks = ctx.div_ceil(BLOCK_SIZE);
        blocks_per_seq.push(n_blocks);
        // Pad to max_blocks_per_seq with the last allocated block (the
        // kernel never reads past num_context_blocks, so the padding
        // value doesn't matter but must be in-bounds).
        let mut row: Vec<u32> = (next_block..next_block + n_blocks).collect();
        while row.len() < max_blocks_per_seq as usize {
            row.push(next_block + n_blocks - 1);
        }
        block_table.extend(row);
        next_block += n_blocks;
    }
    let total_blocks = next_block;

    // Allocate the K/V pool and write each sequence's K/V into the
    // physical blocks it owns. Reuse the single-seq pool builder for
    // each sequence's slice of the global pool.
    let head_per_block_k =
        (HEAD_SIZE as usize / X_PACK as usize) * BLOCK_SIZE as usize * X_PACK as usize;
    let stride_block_k = NUM_KV_HEADS as usize * head_per_block_k;
    let head_per_block_v = HEAD_SIZE as usize * BLOCK_SIZE as usize;
    let stride_block_v = NUM_KV_HEADS as usize * head_per_block_v;
    let mut k_pool = vec![0u16; total_blocks as usize * stride_block_k];
    let mut v_pool = vec![0u16; total_blocks as usize * stride_block_v];

    let mut block_cursor: usize = 0;
    for (s, &ctx) in context_lens.iter().enumerate() {
        let n_blocks = blocks_per_seq[s] as usize;
        let local_k = build_k_pool_bf16(
            &k_per_seq[s],
            ctx as usize,
            HEAD_SIZE as usize,
            NUM_KV_HEADS as usize,
            BLOCK_SIZE as usize,
            n_blocks,
            X_PACK as usize,
        );
        let local_v = build_v_pool_bf16(
            &v_per_seq[s],
            ctx as usize,
            HEAD_SIZE as usize,
            NUM_KV_HEADS as usize,
            BLOCK_SIZE as usize,
            n_blocks,
        );
        let k_dst =
            &mut k_pool[block_cursor * stride_block_k..(block_cursor + n_blocks) * stride_block_k];
        k_dst.copy_from_slice(&local_k);
        let v_dst =
            &mut v_pool[block_cursor * stride_block_v..(block_cursor + n_blocks) * stride_block_v];
        v_dst.copy_from_slice(&local_v);
        block_cursor += n_blocks;
    }

    // Per-query distinct Q vectors (so the kernel must look up the right
    // q_token_idx-th row from the ragged buffer).
    let mut q_host_f32 = vec![0.0f32; total_queries as usize * HEAD_SIZE as usize];
    for q in 0..total_queries as usize {
        for d in 0..HEAD_SIZE as usize {
            q_host_f32[q * HEAD_SIZE as usize + d] = 0.3 + 0.05 * (q as f32) + 0.005 * (d as f32);
        }
    }
    let q_host_bf16: Vec<u16> = q_host_f32.iter().copied().map(f32_to_bf16_bits).collect();

    let key_pool = state.device.new_buffer_with_data(
        k_pool.as_ptr() as *const _,
        std::mem::size_of_val(k_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let value_pool = state.device.new_buffer_with_data(
        v_pool.as_ptr() as *const _,
        std::mem::size_of_val(v_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let q_buf = state.device.new_buffer_with_data(
        q_host_bf16.as_ptr() as *const _,
        std::mem::size_of_val(q_host_bf16.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_table_buf = state.device.new_buffer_with_data(
        block_table.as_ptr() as *const _,
        std::mem::size_of_val(block_table.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_lens_buf = state.device.new_buffer_with_data(
        context_lens.as_ptr() as *const _,
        std::mem::size_of_val(context_lens.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cu_seqlens_q_buf = state.device.new_buffer_with_data(
        cu_seqlens_q.as_ptr() as *const _,
        std::mem::size_of_val(cu_seqlens_q.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();
    let q_stride = (NUM_HEADS * HEAD_SIZE) as i32;
    let kv_block_stride = (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE) as i32;
    let kv_head_stride = (HEAD_SIZE * BLOCK_SIZE) as i32;

    let params = PagedAttentionVarlenParams {
        num_seqs,
        total_queries,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: max_context_len,
        max_num_blocks_per_seq: max_blocks_per_seq,
        scale,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let q_raw = RawBufferInfo {
        ptr: q_buf.as_ptr() as *mut std::ffi::c_void,
        offset: 0,
    };
    let varlen_out = unsafe {
        dispatch_paged_attention_varlen_auto(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table_buf,
            &seq_lens_buf,
            &cu_seqlens_q_buf,
            max_context_len,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("ragged varlen dispatch must succeed");

    let kernel_out = read_output_bf16(
        state,
        &varlen_out.buffer,
        (total_queries * HEAD_SIZE) as usize,
    );

    // Host reference: walk every query token, compute effective_ctx,
    // and reference-attend over the matching sequence's K/V.
    let tol = 5.0e-2_f32;
    for q_token_idx in 0..total_queries as usize {
        // Find sequence index (mirrors the kernel's binary search).
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

        // Round Q to bf16 then back so the host ref sees the same input
        // the kernel sees (Q is stored as bf16 in q_buf).
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
    let state = MetalState::get().expect("Metal must be available");

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;
    const X_PACK: u32 = 8;

    let q_lens = [1u32, 3];
    let context_lens = [1024u32, 1024];
    let cu_seqlens_q: Vec<i32> = vec![0, 1, 4];
    let total_queries: u32 = q_lens.iter().sum();
    let num_seqs: u32 = q_lens.len() as u32;
    let max_context_len: u32 = *context_lens.iter().max().unwrap();
    let max_blocks_per_seq: u32 = max_context_len.div_ceil(BLOCK_SIZE);

    let mut k_per_seq: Vec<Vec<f32>> = Vec::with_capacity(num_seqs as usize);
    let mut v_per_seq: Vec<Vec<f32>> = Vec::with_capacity(num_seqs as usize);
    for (s, &ctx) in context_lens.iter().enumerate() {
        let (k, v) = build_synthetic_kv(ctx as usize, HEAD_SIZE as usize, s as f32 + 2.0);
        k_per_seq.push(k);
        v_per_seq.push(v);
    }

    let mut block_table: Vec<u32> = Vec::new();
    let mut next_block: u32 = 0;
    for &ctx in &context_lens {
        let n = ctx.div_ceil(BLOCK_SIZE);
        let mut row: Vec<u32> = (next_block..next_block + n).collect();
        while row.len() < max_blocks_per_seq as usize {
            row.push(next_block + n - 1);
        }
        block_table.extend(row);
        next_block += n;
    }
    let total_blocks = next_block;

    let head_per_block_k =
        (HEAD_SIZE as usize / X_PACK as usize) * BLOCK_SIZE as usize * X_PACK as usize;
    let stride_block_k = NUM_KV_HEADS as usize * head_per_block_k;
    let head_per_block_v = HEAD_SIZE as usize * BLOCK_SIZE as usize;
    let stride_block_v = NUM_KV_HEADS as usize * head_per_block_v;
    let mut k_pool = vec![0u16; total_blocks as usize * stride_block_k];
    let mut v_pool = vec![0u16; total_blocks as usize * stride_block_v];
    let mut block_cursor: usize = 0;
    for (s, &ctx) in context_lens.iter().enumerate() {
        let n = ctx.div_ceil(BLOCK_SIZE) as usize;
        let local_k = build_k_pool_bf16(
            &k_per_seq[s],
            ctx as usize,
            HEAD_SIZE as usize,
            NUM_KV_HEADS as usize,
            BLOCK_SIZE as usize,
            n,
            X_PACK as usize,
        );
        let local_v = build_v_pool_bf16(
            &v_per_seq[s],
            ctx as usize,
            HEAD_SIZE as usize,
            NUM_KV_HEADS as usize,
            BLOCK_SIZE as usize,
            n,
        );
        k_pool[block_cursor * stride_block_k..(block_cursor + n) * stride_block_k]
            .copy_from_slice(&local_k);
        v_pool[block_cursor * stride_block_v..(block_cursor + n) * stride_block_v]
            .copy_from_slice(&local_v);
        block_cursor += n;
    }

    let mut q_host_f32 = vec![0.0f32; total_queries as usize * HEAD_SIZE as usize];
    for q in 0..total_queries as usize {
        for d in 0..HEAD_SIZE as usize {
            q_host_f32[q * HEAD_SIZE as usize + d] = 0.2 + 0.07 * (q as f32) + 0.003 * (d as f32);
        }
    }
    let q_host_bf16: Vec<u16> = q_host_f32.iter().copied().map(f32_to_bf16_bits).collect();

    let key_pool = state.device.new_buffer_with_data(
        k_pool.as_ptr() as *const _,
        std::mem::size_of_val(k_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let value_pool = state.device.new_buffer_with_data(
        v_pool.as_ptr() as *const _,
        std::mem::size_of_val(v_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let q_buf = state.device.new_buffer_with_data(
        q_host_bf16.as_ptr() as *const _,
        std::mem::size_of_val(q_host_bf16.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_table_buf = state.device.new_buffer_with_data(
        block_table.as_ptr() as *const _,
        std::mem::size_of_val(block_table.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_lens_buf = state.device.new_buffer_with_data(
        context_lens.as_ptr() as *const _,
        std::mem::size_of_val(context_lens.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cu_seqlens_q_buf = state.device.new_buffer_with_data(
        cu_seqlens_q.as_ptr() as *const _,
        std::mem::size_of_val(cu_seqlens_q.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();
    let q_stride = (NUM_HEADS * HEAD_SIZE) as i32;
    let kv_block_stride = (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE) as i32;
    let kv_head_stride = (HEAD_SIZE * BLOCK_SIZE) as i32;

    let params = PagedAttentionVarlenParams {
        num_seqs,
        total_queries,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: max_context_len,
        max_num_blocks_per_seq: max_blocks_per_seq,
        scale,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let q_raw = RawBufferInfo {
        ptr: q_buf.as_ptr() as *mut std::ffi::c_void,
        offset: 0,
    };
    let varlen_out = unsafe {
        dispatch_paged_attention_varlen_auto(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table_buf,
            &seq_lens_buf,
            &cu_seqlens_q_buf,
            max_context_len,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("v2 ragged dispatch must succeed");

    let kernel_out = read_output_bf16(
        state,
        &varlen_out.buffer,
        (total_queries * HEAD_SIZE) as usize,
    );

    // V2 stores per-partition outputs in bf16 (io_dtype), then the reduce
    // kernel re-weights them in fp32 and writes bf16 again. With 1024
    // context length the bf16 round-trip on partition outputs caps
    // achievable precision to ~1% relative — looser than the V1 5e-2
    // absolute tolerance because the output magnitudes here are larger
    // (~25-50 vs ~3 in the V1 test). Use a relative-or-absolute tolerance
    // that matches what the single-row V2 path also achieves.
    let rel_tol = 1.5e-2_f32;
    let abs_tol = 5.0e-2_f32;
    for q_token_idx in 0..total_queries as usize {
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
// So they attend to overlapping but distinct KV windows. With distinct
// Q vectors the outputs must differ, AND the q=0 output must NOT depend
// on KV[31] (the last token) — a kernel that ignores the per-query
// cutoff would fold KV[31] into q=0's output and the cross-check fails.
//
// We verify (a) the two outputs differ, and (b) the q=0 output matches
// the host reference that masks KV[31].
// =============================================================================

#[test]
fn varlen_per_query_causal_cutoff_is_enforced() {
    if !metal_available() {
        eprintln!("skipping varlen_per_query_causal_cutoff_is_enforced: Metal unavailable");
        return;
    }
    let state = MetalState::get().expect("Metal must be available");

    const HEAD_SIZE: u32 = 64;
    const NUM_HEADS: u32 = 1;
    const NUM_KV_HEADS: u32 = 1;
    const BLOCK_SIZE: u32 = 16;
    const CONTEXT_LEN: u32 = 32;
    const NUM_BLOCKS: u32 = 2;
    const X_PACK: u32 = 8;

    let (k_per_token, v_per_token) =
        build_synthetic_kv(CONTEXT_LEN as usize, HEAD_SIZE as usize, 3.0);
    // Make KV[31] very large in V so a kernel that incorrectly
    // includes it in q=0's attention would shift the output noticeably.
    let mut v_per_token = v_per_token;
    for d in 0..HEAD_SIZE as usize {
        v_per_token[31 * HEAD_SIZE as usize + d] = 100.0 + d as f32;
    }

    let mut q_host_f32 = vec![0.0f32; 2 * HEAD_SIZE as usize];
    for d in 0..HEAD_SIZE as usize {
        q_host_f32[d] = 0.5 + 0.01 * d as f32;
        q_host_f32[HEAD_SIZE as usize + d] = -0.5 + 0.02 * d as f32;
    }
    let q_bf16: Vec<u16> = q_host_f32.iter().copied().map(f32_to_bf16_bits).collect();

    let k_pool = build_k_pool_bf16(
        &k_per_token,
        CONTEXT_LEN as usize,
        HEAD_SIZE as usize,
        NUM_KV_HEADS as usize,
        BLOCK_SIZE as usize,
        NUM_BLOCKS as usize,
        X_PACK as usize,
    );
    let v_pool = build_v_pool_bf16(
        &v_per_token,
        CONTEXT_LEN as usize,
        HEAD_SIZE as usize,
        NUM_KV_HEADS as usize,
        BLOCK_SIZE as usize,
        NUM_BLOCKS as usize,
    );

    let cu_seqlens_q: Vec<i32> = vec![0, 2];
    let block_table: Vec<u32> = vec![0, 1];
    let seq_lens: Vec<u32> = vec![CONTEXT_LEN];

    let key_pool = state.device.new_buffer_with_data(
        k_pool.as_ptr() as *const _,
        std::mem::size_of_val(k_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let value_pool = state.device.new_buffer_with_data(
        v_pool.as_ptr() as *const _,
        std::mem::size_of_val(v_pool.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let q_buf = state.device.new_buffer_with_data(
        q_bf16.as_ptr() as *const _,
        std::mem::size_of_val(q_bf16.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_table_buf = state.device.new_buffer_with_data(
        block_table.as_ptr() as *const _,
        std::mem::size_of_val(block_table.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_lens_buf = state.device.new_buffer_with_data(
        seq_lens.as_ptr() as *const _,
        std::mem::size_of_val(seq_lens.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cu_seqlens_q_buf = state.device.new_buffer_with_data(
        cu_seqlens_q.as_ptr() as *const _,
        std::mem::size_of_val(cu_seqlens_q.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();
    let q_stride = (NUM_HEADS * HEAD_SIZE) as i32;
    let kv_block_stride = (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE) as i32;
    let kv_head_stride = (HEAD_SIZE * BLOCK_SIZE) as i32;

    let params = PagedAttentionVarlenParams {
        num_seqs: 1,
        total_queries: 2,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: CONTEXT_LEN,
        max_num_blocks_per_seq: block_table.len() as u32,
        scale,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let q_raw = RawBufferInfo {
        ptr: q_buf.as_ptr() as *mut std::ffi::c_void,
        offset: 0,
    };
    let out = unsafe {
        dispatch_paged_attention_varlen_v1_raw(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table_buf,
            &seq_lens_buf,
            &cu_seqlens_q_buf,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("varlen dispatch must succeed");

    let kernel_out = read_output_bf16(state, &out.buffer, (2 * HEAD_SIZE) as usize);
    let q0_out = &kernel_out[0..HEAD_SIZE as usize];
    let q1_out = &kernel_out[HEAD_SIZE as usize..2 * HEAD_SIZE as usize];

    // The two outputs MUST differ — distinct Q vectors and distinct
    // effective_ctx ensure they sample different distributions.
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

    // q=0 reference: effective_ctx = 31, so KV[31] must be excluded.
    // If the kernel mistakenly attends to KV[31] (V=100+) the output
    // would be dragged up significantly.
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

    // Cross-check: if we accidentally attend to KV[31] for q=0, the
    // output would be at least ~5 in magnitude (V[31] is 100+, softmax
    // weight ~0.03 for one of 32 positions). Confirm q0 stays below
    // that magnitude bound.
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
