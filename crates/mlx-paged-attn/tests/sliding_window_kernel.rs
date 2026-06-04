//! Sliding-window kernel test.
//!
//! End-to-end check that the `sliding_window` parameter on the
//! `paged_attention` Metal kernel actually masks K positions older than
//! `context_len - sliding_window` from the softmax computation.
//!
//! Strategy: drive `dispatch_paged_attention_v1_raw` directly with a
//! deterministic K/V pool and a constant Q, then compute a host-side
//! reference using the same softmax-over-K-V formula the kernel
//! implements. Any divergence between the kernel output and the host
//! reference (within a small bf16 tolerance) is a bug. We test 5
//! distinct window sizes (including 0 = no mask) plus a degenerate
//! window > context_len case (which must collapse to the no-mask
//! behaviour).
//!
//! Skips cleanly on hosts without Metal.

#![cfg(target_os = "macos")]

use metal::MTLResourceOptions;
use metal::foreign_types::ForeignType;
use mlx_paged_attn::metal::{
    MetalDtype, MetalState, PagedAttentionParams, RawBufferInfo, dispatch_paged_attention_v1_raw,
};

// Test config: 1 sequence, 1 KV head, 1 query head, head_size 64.
// Context length = 32 tokens (≤ PARTITION_SIZE so V1 path runs).
// Block size 16 → 2 blocks per sequence.
const NUM_SEQS: u32 = 1;
const NUM_HEADS: u32 = 1;
const NUM_KV_HEADS: u32 = 1;
const HEAD_SIZE: u32 = 64;
const BLOCK_SIZE: u32 = 16;
const CONTEXT_LEN: u32 = 32;
const NUM_BLOCKS: u32 = 2;
const X_PACK: u32 = 8; // bf16 → x = 16 / sizeof(bf16) = 8

/// Convert f32 → bf16 (16-bit truncation of the upper bits).
fn f32_to_bf16_bits(x: f32) -> u16 {
    // Round-to-nearest-even using the next bit of the mantissa.
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// True iff Metal is reachable on this host.
fn metal_available() -> bool {
    MetalState::get().is_ok()
}

/// Compute the host-side reference paged-attention output for the K/V
/// pool, query, and sliding_window combination used in the kernel test.
///
/// Mirrors the kernel's softmax-over-K-V exactly:
///   * scores[t] = sum_d Q[d] * K[t][d] * scale
///   * mask[t] = (t < context_len) AND (sliding_window == 0 OR
///     t >= context_len - sliding_window)
///   * masked_scores[t] = mask[t] ? scores[t] : -INFINITY
///   * weights = softmax(masked_scores)
///   * output[d] = sum_t weights[t] * V[t][d]
///
/// `k_per_token` and `v_per_token` are flat row-major K/V stored as
/// `[context_len][head_size]` f32.
fn host_reference_attention(
    q: &[f32],           // [head_size]
    k_per_token: &[f32], // [context_len * head_size]
    v_per_token: &[f32], // [context_len * head_size]
    context_len: usize,
    head_size: usize,
    scale: f32,
    sliding_window: i32,
) -> Vec<f32> {
    assert_eq!(q.len(), head_size);
    assert_eq!(k_per_token.len(), context_len * head_size);
    assert_eq!(v_per_token.len(), context_len * head_size);

    let sw = sliding_window as i64;
    let lower_bound: i64 = if sw > 0 && (context_len as i64) > sw {
        context_len as i64 - sw
    } else {
        0
    };

    // 1. scores
    let mut scores = vec![0.0f32; context_len];
    for t in 0..context_len {
        let mut acc = 0.0f32;
        for d in 0..head_size {
            acc += q[d] * k_per_token[t * head_size + d];
        }
        scores[t] = acc * scale;
    }

    // 2. apply mask (set masked → -inf so softmax weight is 0).
    for (t, score) in scores.iter_mut().enumerate() {
        let is_masked = sw > 0 && (t as i64) < lower_bound;
        if is_masked {
            *score = f32::NEG_INFINITY;
        }
    }

    // 3. softmax (numerically stable).
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut exps = vec![0.0f32; context_len];
    for (i, e) in exps.iter_mut().enumerate() {
        *e = (scores[i] - max_score).exp();
        sum_exp += *e;
    }
    let inv_sum = 1.0f32 / (sum_exp + 1e-6);
    for e in exps.iter_mut() {
        *e *= inv_sum;
    }

    // 4. weighted sum of V
    let mut out = vec![0.0f32; head_size];
    for d in 0..head_size {
        let mut acc = 0.0f32;
        for t in 0..context_len {
            acc += exps[t] * v_per_token[t * head_size + d];
        }
        out[d] = acc;
    }
    out
}

/// Build the K-pool buffer in the layout
/// `[num_blocks, num_kv_heads, head_size/x_pack, block_size, x_pack]`
/// (kernel-native bf16) from a flat `[context_len][head_size]` source.
///
/// Slot t lives in block (t / BLOCK_SIZE), block_offset (t % BLOCK_SIZE).
fn build_k_pool_bf16(k_per_token: &[f32]) -> Vec<u16> {
    let num_blocks = NUM_BLOCKS as usize;
    let num_kv_heads = NUM_KV_HEADS as usize;
    let head_size = HEAD_SIZE as usize;
    let block_size = BLOCK_SIZE as usize;
    let x_pack = X_PACK as usize;
    let head_per_block = (head_size / x_pack) * block_size * x_pack;
    let stride_block = num_kv_heads * head_per_block;
    let stride_head = head_per_block;
    let stride_xidx = block_size * x_pack;
    let stride_blockoff = x_pack;

    let mut pool = vec![0u16; num_blocks * stride_block];

    let context_len = CONTEXT_LEN as usize;
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
                pool[pool_idx] = f32_to_bf16_bits(k_per_token[t * head_size + j]);
            }
        }
    }
    pool
}

/// Build the V-pool buffer in the layout
/// `[num_blocks, num_kv_heads, head_size, block_size]` from a flat
/// `[context_len][head_size]` source.
fn build_v_pool_bf16(v_per_token: &[f32]) -> Vec<u16> {
    let num_blocks = NUM_BLOCKS as usize;
    let num_kv_heads = NUM_KV_HEADS as usize;
    let head_size = HEAD_SIZE as usize;
    let block_size = BLOCK_SIZE as usize;
    let head_per_block = head_size * block_size;
    let stride_block = num_kv_heads * head_per_block;
    let stride_head = head_per_block;
    let stride_j = block_size;

    let mut pool = vec![0u16; num_blocks * stride_block];

    let context_len = CONTEXT_LEN as usize;
    for t in 0..context_len {
        let block_idx = t / block_size;
        let block_offset = t % block_size;
        for h in 0..num_kv_heads {
            for j in 0..head_size {
                let pool_idx =
                    block_idx * stride_block + h * stride_head + j * stride_j + block_offset;
                pool[pool_idx] = f32_to_bf16_bits(v_per_token[t * head_size + j]);
            }
        }
    }
    pool
}

/// Run one paged_attention dispatch via the Rust raw entry point and
/// return the output as a flat Vec<f32> of length `head_size`. The
/// caller controls only `sliding_window`; everything else is fixed by
/// the test config.
#[allow(clippy::too_many_arguments)]
fn run_dispatch(
    state: &MetalState,
    k_pool_bytes: &[u16],
    v_pool_bytes: &[u16],
    q_bf16: &[u16],
    block_table: &[u32],
    seq_lens: &[u32],
    sliding_window: i32,
    scale: f32,
) -> Vec<f32> {
    let key_pool = state.device.new_buffer_with_data(
        k_pool_bytes.as_ptr() as *const _,
        std::mem::size_of_val(k_pool_bytes) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let value_pool = state.device.new_buffer_with_data(
        v_pool_bytes.as_ptr() as *const _,
        std::mem::size_of_val(v_pool_bytes) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let q_buf = state.device.new_buffer_with_data(
        q_bf16.as_ptr() as *const _,
        std::mem::size_of_val(q_bf16) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_table_buf = state.device.new_buffer_with_data(
        block_table.as_ptr() as *const _,
        std::mem::size_of_val(block_table) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_lens_buf = state.device.new_buffer_with_data(
        seq_lens.as_ptr() as *const _,
        std::mem::size_of_val(seq_lens) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let q_stride = (NUM_HEADS * HEAD_SIZE) as i32;
    let kv_block_stride = (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE) as i32;
    let kv_head_stride = (HEAD_SIZE * BLOCK_SIZE) as i32;

    let max_blocks_per_seq = block_table.len() as u32 / NUM_SEQS;
    let params = PagedAttentionParams {
        num_seqs: NUM_SEQS,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: CONTEXT_LEN,
        max_num_blocks_per_seq: max_blocks_per_seq,
        scale,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window,
    };

    let q_raw = RawBufferInfo {
        ptr: q_buf.as_ptr() as *mut std::ffi::c_void,
        offset: 0,
    };

    let out = unsafe {
        dispatch_paged_attention_v1_raw(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table_buf,
            &seq_lens_buf,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("dispatch_paged_attention_v1_raw must succeed");

    // Read the output buffer back as bf16 → f32.
    let out_elems = (NUM_SEQS * NUM_HEADS * HEAD_SIZE) as usize;
    let out_bytes = out_elems * std::mem::size_of::<u16>();
    // Output buffer is StorageModePrivate; blit it to a shared buffer
    // first so we can read it host-side.
    let shared_out = state
        .device
        .new_buffer(out_bytes as u64, MTLResourceOptions::StorageModeShared);
    let cmd = state.command_queue.new_command_buffer();
    let blit = cmd.new_blit_command_encoder();
    blit.copy_from_buffer(&out.buffer, 0, &shared_out, 0, out_bytes as u64);
    blit.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out_bits =
        unsafe { std::slice::from_raw_parts(shared_out.contents() as *const u16, out_elems) };
    out_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect()
}

/// Build deterministic K/V values for the test. K[t][d] depends on (t,
/// d) so each token has a different attention score against a constant
/// Q; V[t][d] depends on t so the weighted sum reveals which tokens
/// contributed.
fn build_synthetic_kv() -> (Vec<f32>, Vec<f32>) {
    let context_len = CONTEXT_LEN as usize;
    let head_size = HEAD_SIZE as usize;
    let mut k = vec![0.0f32; context_len * head_size];
    let mut v = vec![0.0f32; context_len * head_size];

    for t in 0..context_len {
        for d in 0..head_size {
            // K: small magnitudes so QK · scale ≈ ±2 (reasonable softmax range)
            k[t * head_size + d] = ((t as f32) * 0.1 + (d as f32) * 0.01).sin() * 0.5;
            // V: each token contributes a distinct constant vector so the
            // weighted sum reveals which tokens dominated.
            v[t * head_size + d] = (t as f32 + 1.0) * 0.1 + (d as f32) * 0.001;
        }
    }
    (k, v)
}

/// Constant query: every dim = 1.0.
fn build_synthetic_q() -> Vec<f32> {
    vec![1.0f32; HEAD_SIZE as usize]
}

/// Run the kernel + reference for one sliding_window value and assert
/// they agree within bf16 tolerance.
fn check_sliding_window(window: i32, label: &str) {
    if !metal_available() {
        eprintln!("skipping {label}: Metal not available on this host");
        return;
    }
    let state = MetalState::get().expect("Metal must be available (checked above)");

    let (k_per_token, v_per_token) = build_synthetic_kv();
    let q = build_synthetic_q();
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();

    let q_bf16: Vec<u16> = q.iter().copied().map(f32_to_bf16_bits).collect();
    let k_pool_bf16 = build_k_pool_bf16(&k_per_token);
    let v_pool_bf16 = build_v_pool_bf16(&v_per_token);

    // Block table: [block_0=0, block_1=1] for our 2-block sequence.
    let block_table: Vec<u32> = vec![0, 1];
    let seq_lens: Vec<u32> = vec![CONTEXT_LEN];

    let kernel_out = run_dispatch(
        state,
        &k_pool_bf16,
        &v_pool_bf16,
        &q_bf16,
        &block_table,
        &seq_lens,
        window,
        scale,
    );
    let host_ref = host_reference_attention(
        &q,
        &k_per_token,
        &v_per_token,
        CONTEXT_LEN as usize,
        HEAD_SIZE as usize,
        scale,
        window,
    );

    // Tolerance: bf16 has ~7-bit mantissa → ~1e-2 relative precision.
    // The accumulator inside the kernel is f32, so the dominant error
    // comes from truncating Q, K, V to bf16. A 5e-2 absolute tolerance
    // covers ~3 ULPs at typical magnitudes for the synthetic V values
    // (|V| ≤ 3.4 in this config).
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    for d in 0..(HEAD_SIZE as usize) {
        let diff = (kernel_out[d] - host_ref[d]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = d;
        }
    }
    let tol = 5.0e-2_f32;
    assert!(
        max_diff < tol,
        "{label}: kernel output diverges from host reference at dim={max_diff_idx} \
         (kernel={}, ref={}, diff={max_diff} > tol={tol})",
        kernel_out[max_diff_idx],
        host_ref[max_diff_idx]
    );
}

/// Sanity check: with sliding_window=0 the kernel must reproduce the
/// full-context (unmasked) behaviour.
#[test]
fn sliding_window_zero_matches_full_context() {
    check_sliding_window(0, "sliding_window_zero_matches_full_context");
}

/// Mid-sized window: 16 = half the context. The host reference
/// independently masks tokens [0..16) and the kernel must agree.
#[test]
fn sliding_window_16_masks_first_half() {
    check_sliding_window(16, "sliding_window_16_masks_first_half");
}

/// Small window: 8 = quarter of the context.
#[test]
fn sliding_window_8_masks_three_quarters() {
    check_sliding_window(8, "sliding_window_8_masks_three_quarters");
}

/// Single-token window: only the last token contributes.
#[test]
fn sliding_window_one_keeps_only_last_token() {
    check_sliding_window(1, "sliding_window_one_keeps_only_last_token");
}

/// Window > context_len collapses to no mask. (lower_bound = 0).
#[test]
fn sliding_window_larger_than_context_collapses_to_no_mask() {
    check_sliding_window(
        (CONTEXT_LEN as i32) + 100,
        "sliding_window_larger_than_context_collapses_to_no_mask",
    );
}

/// 5th window size — exactly equal to context (lower_bound = 0, no
/// masking). Documents the boundary case.
#[test]
fn sliding_window_equal_to_context_keeps_all() {
    check_sliding_window(
        CONTEXT_LEN as i32,
        "sliding_window_equal_to_context_keeps_all",
    );
}

/// Cross-check: window=1 must produce a STRICTLY DIFFERENT output from
/// window=0 (otherwise the mask is silently a no-op).
#[test]
fn sliding_window_one_differs_from_full_context() {
    if !metal_available() {
        eprintln!("skipping sliding_window_one_differs_from_full_context: Metal unavailable");
        return;
    }
    let state = MetalState::get().expect("Metal must be available (checked above)");

    let (k_per_token, v_per_token) = build_synthetic_kv();
    let q = build_synthetic_q();
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();

    let q_bf16: Vec<u16> = q.iter().copied().map(f32_to_bf16_bits).collect();
    let k_pool_bf16 = build_k_pool_bf16(&k_per_token);
    let v_pool_bf16 = build_v_pool_bf16(&v_per_token);
    let block_table: Vec<u32> = vec![0, 1];
    let seq_lens: Vec<u32> = vec![CONTEXT_LEN];

    let out_no_mask = run_dispatch(
        state,
        &k_pool_bf16,
        &v_pool_bf16,
        &q_bf16,
        &block_table,
        &seq_lens,
        0,
        scale,
    );
    let out_w1 = run_dispatch(
        state,
        &k_pool_bf16,
        &v_pool_bf16,
        &q_bf16,
        &block_table,
        &seq_lens,
        1,
        scale,
    );

    let max_diff = out_no_mask
        .iter()
        .zip(out_w1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1.0e-3,
        "sliding_window=1 must differ from sliding_window=0 (max_diff={max_diff}); \
         a no-op mask would silently produce equal outputs and pass the host-ref \
         check via shared bias"
    );
}
