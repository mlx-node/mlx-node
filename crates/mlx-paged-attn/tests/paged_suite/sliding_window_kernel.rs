//! Sliding-window kernel test.
//!
//! End-to-end check that the `sliding_window` parameter on the
//! `paged_attention` Metal kernel actually masks K positions older than
//! `context_len - sliding_window` from the softmax computation.
//!
//! Strategy: populate a `LayerKVPool` via `mlx_sys::mlx_paged_kv_write_forward`
//! and drive `mlx_sys::mlx_paged_attention_forward` with a deterministic K/V
//! pool and a constant Q, then compute a host-side reference using the same
//! softmax-over-K-V formula the kernel implements. Any divergence between the
//! kernel output and the host reference (within a small bf16 tolerance) is a
//! bug. We test 6 distinct window sizes on the V1 path (including 0 = no mask
//! and window > context_len) plus a V2 high-partition sliding-window case.
//!
//! Skips cleanly on hosts without Metal.

use mlx_paged_attn::metal::{MetalDtype, MetalState};
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::mlx_array;

// V1 test config: 1 sequence, 1 KV head, 1 query head, head_size 64.
// Context length = 32 tokens (<= PARTITION_SIZE so V1 path runs).
// Block size 16 -> 2 blocks per sequence.
const NUM_HEADS: u32 = 1;
const NUM_KV_HEADS: u32 = 1;
const HEAD_SIZE: u32 = 64;
const BLOCK_SIZE: u32 = 16;
const CONTEXT_LEN: u32 = 32;
const KV_DTYPE_BF16: u8 = 1;

struct Owned(*mut mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

/// Convert f32 -> bf16 (round-to-nearest-even).
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
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

    // 2. apply mask (set masked -> -inf so softmax weight is 0).
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

/// Write `k_per_token` / `v_per_token` into a fresh BF16 `LayerKVPool` via
/// `mlx_paged_kv_write_forward` and dispatch `mlx_paged_attention_forward`
/// with the requested `sliding_window`.
fn run_dispatch(
    k_per_token: &[f32],
    v_per_token: &[f32],
    q: &[f32],
    context_len: u32,
    sliding_window: i32,
    scale: f32,
) -> Vec<f32> {
    let num_blocks = context_len.div_ceil(BLOCK_SIZE);
    let config = PagedAttentionConfig {
        block_size: BLOCK_SIZE,
        gpu_memory_mb: 256,
        head_size: HEAD_SIZE,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: 1,
        max_seq_len: Some(num_blocks * BLOCK_SIZE),
        max_batch_size: Some(1),
    };
    let pool = LayerKVPool::new(config, num_blocks, num_blocks, MetalDtype::BFloat16)
        .expect("LayerKVPool::new must succeed");

    let k_bf16: Vec<u16> = k_per_token.iter().copied().map(f32_to_bf16_bits).collect();
    let v_bf16: Vec<u16> = v_per_token.iter().copied().map(f32_to_bf16_bits).collect();
    let q_bf16: Vec<u16> = q.iter().copied().map(f32_to_bf16_bits).collect();

    let kv_shape = [context_len as i64, NUM_KV_HEADS as i64, HEAD_SIZE as i64];
    let new_k = from_bf16(&k_bf16, &kv_shape);
    let new_v = from_bf16(&v_bf16, &kv_shape);
    let slots: Vec<i64> = (0..context_len as i64).collect();
    let slot_mapping = from_i64(&slots);
    let k_pool_in = Owned(pool.key_cache_array_raw(0).expect("key_cache_array_raw"));
    let v_pool_in = Owned(
        pool.value_cache_array_raw(0)
            .expect("value_cache_array_raw"),
    );
    let k_scale_in = from_f32(&[1.0], &[1]);
    let v_scale_in = from_f32(&[1.0], &[1]);

    let mut write_out = [std::ptr::null_mut::<mlx_array>(); 4];
    let [k_out, v_out, ks_out, vs_out] = &mut write_out;
    let ok = unsafe {
        mlx_sys::mlx_paged_kv_write_forward(
            k_pool_in.0,
            v_pool_in.0,
            new_k.0,
            new_v.0,
            slot_mapping.0,
            k_scale_in.0,
            v_scale_in.0,
            BLOCK_SIZE as i32,
            NUM_KV_HEADS as i32,
            HEAD_SIZE as i32,
            KV_DTYPE_BF16,
            false,
            k_out,
            v_out,
            ks_out,
            vs_out,
        )
    };
    assert!(ok, "mlx_paged_kv_write_forward must succeed");
    let written = write_out.map(Owned);
    let mut write_handles = write_out;
    assert!(unsafe { mlx_sys::mlx_eval(write_handles.as_mut_ptr(), 4) });
    let [k_pool, v_pool, k_scale, v_scale] = written;

    let block_table_vec: Vec<i32> = (0..num_blocks as i32).collect();
    let block_table = from_i32(&block_table_vec, &[1, num_blocks as i64]);
    let seq_lens = from_i32(&[context_len as i32], &[1]);
    let q_arr = from_bf16(&q_bf16, &[1, NUM_HEADS as i64, HEAD_SIZE as i64]);

    let out = unsafe {
        mlx_sys::mlx_paged_attention_forward(
            q_arr.0,
            k_pool.0,
            v_pool.0,
            block_table.0,
            seq_lens.0,
            k_scale.0,
            v_scale.0,
            scale,
            /* softcap */ 0.0,
            sliding_window,
            BLOCK_SIZE as i32,
            NUM_HEADS as i32,
            NUM_KV_HEADS as i32,
            HEAD_SIZE as i32,
            KV_DTYPE_BF16,
        )
    };
    assert!(!out.is_null(), "mlx_paged_attention_forward must succeed");
    let out = Owned(out);
    let mut out_handles = [out.0];
    assert!(unsafe { mlx_sys::mlx_eval(out_handles.as_mut_ptr(), 1) });
    to_f32(&out)
}

/// Build deterministic K/V values for the test. K[t][d] depends on (t,
/// d) so each token has a different attention score against a constant
/// Q; V[t][d] depends on t so the weighted sum reveals which tokens
/// contributed.
fn build_synthetic_kv(context_len: usize) -> (Vec<f32>, Vec<f32>) {
    let head_size = HEAD_SIZE as usize;
    let mut k = vec![0.0f32; context_len * head_size];
    let mut v = vec![0.0f32; context_len * head_size];

    for t in 0..context_len {
        for d in 0..head_size {
            // K: small magnitudes so QK · scale ~= +/-2 (reasonable softmax range)
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

    let (k_per_token, v_per_token) = build_synthetic_kv(CONTEXT_LEN as usize);
    let q = build_synthetic_q();
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();

    let kernel_out = run_dispatch(&k_per_token, &v_per_token, &q, CONTEXT_LEN, window, scale);
    let host_ref = host_reference_attention(
        &q,
        &k_per_token,
        &v_per_token,
        CONTEXT_LEN as usize,
        HEAD_SIZE as usize,
        scale,
        window,
    );

    // Tolerance: bf16 has ~7-bit mantissa -> ~1e-2 relative precision.
    // The accumulator inside the kernel is f32, so the dominant error
    // comes from truncating Q, K, V to bf16. A 5e-2 absolute tolerance
    // covers ~3 ULPs at typical magnitudes for the synthetic V values
    // (|V| <= 3.4 in this config).
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

    let (k_per_token, v_per_token) = build_synthetic_kv(CONTEXT_LEN as usize);
    let q = build_synthetic_q();
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();

    let out_no_mask = run_dispatch(&k_per_token, &v_per_token, &q, CONTEXT_LEN, 0, scale);
    let out_w1 = run_dispatch(&k_per_token, &v_per_token, &q, CONTEXT_LEN, 1, scale);

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

/// V2 high-partition path (`context_len > 512`): sliding window cuts across the
/// first partition so partition 0 is wholly masked while partition 1 survives.
#[test]
fn sliding_window_v2_high_partition_matches_host_reference() {
    if !metal_available() {
        eprintln!(
            "skipping sliding_window_v2_high_partition_matches_host_reference: Metal unavailable"
        );
        return;
    }

    const V2_CONTEXT_LEN: u32 = 640;
    const WINDOW: i32 = 96;
    let (k_per_token, v_per_token) = build_synthetic_kv(V2_CONTEXT_LEN as usize);
    let q = build_synthetic_q();
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();

    let kernel_out = run_dispatch(
        &k_per_token,
        &v_per_token,
        &q,
        V2_CONTEXT_LEN,
        WINDOW,
        scale,
    );
    let host_ref = host_reference_attention(
        &q,
        &k_per_token,
        &v_per_token,
        V2_CONTEXT_LEN as usize,
        HEAD_SIZE as usize,
        scale,
        WINDOW,
    );

    let rel_tol = 1.5e-2_f32;
    let abs_tol = 5.0e-2_f32;
    for d in 0..(HEAD_SIZE as usize) {
        let diff = (kernel_out[d] - host_ref[d]).abs();
        let tol = abs_tol.max(rel_tol * host_ref[d].abs());
        assert!(
            diff < tol,
            "V2 sliding_window={WINDOW} diverges at dim={d} \
             (kernel={}, ref={}, diff={diff} > tol={tol})",
            kernel_out[d],
            host_ref[d]
        );
    }
}
