//! P1C-4 — Toy single-layer end-to-end validation of `PagedKVCacheAdapter`.
//!
//! Builds a minimal attention layer end-to-end through the adapter
//! (`update_keys_values` + `gather_kv_for_decode`) and compares the output
//! against a reference computed via MLX's `scaled_dot_product_attention`
//! over a flat `[batch, n_heads, seq_len, head_dim]` K/V layout. This is
//! the validation gate the previous P1C tasks (lifecycle / write / read)
//! never exercised together — each prior test isolated one piece.
//!
//! Three test variants:
//! 1. `test_adapter_e2e_matches_flat_attention_fp16` — Float16 path.
//! 2. `test_adapter_e2e_matches_flat_attention_bf16` — BFloat16 path.
//! 3. `test_adapter_e2e_prefix_reuse_matches_no_reuse` — same K/V replayed
//!    with the prefix-cache hit path; output must equal the no-reuse run.
//!
//! All three gracefully skip on no-Metal hosts (sandboxed CI VMs lack a
//! Metal device; `LayerKVPool::new` returns `Err("No Metal device …")`
//! there). On Metal hardware they run sub-second — no model loading, no
//! network, fixed-size synthetic K/V/Q.
//!
//! Compiled only on macOS — `mlx_paged_attn::metal::MetalDtype` is gated
//! to `target_os = "macos"`, so the test wouldn't even link on Linux. The
//! `#![cfg]` keeps `cargo test` on non-Metal hosts a no-op rather than a
//! compilation error.

#![cfg(target_os = "macos")]

use std::sync::{Arc, Mutex};

use half::{bf16, f16};

use mlx_core::array::{DType, MxArray, scaled_dot_product_attention, synchronize};
use mlx_core::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig, metal::MetalDtype};

// ---------------------------------------------------------------------------
// Shared toy parameters
// ---------------------------------------------------------------------------

const NUM_KV_HEADS: u32 = 4;
const NUM_QUERY_HEADS: u32 = 4; // No GQA — keeps the reference path simple.
const HEAD_SIZE: u32 = 64;
const BLOCK_SIZE: u32 = 8;
const NUM_BLOCKS: u32 = 8;
const NUM_TOKENS: u32 = 24; // Exactly 3 blocks of 8.

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Build the adapter + its (allocator, pool) backing for one run.
/// Returns `None` on no-Metal hosts so the caller can skip cleanly.
fn try_make_fixture(cache_dtype: MetalDtype) -> Option<Fixture> {
    let cfg = PagedAttentionConfig {
        block_size: BLOCK_SIZE,
        gpu_memory_mb: 256, // Unused for our explicit `LayerKVPool::new(num_blocks=...)` call.
        head_size: HEAD_SIZE,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: 1,
        use_fp8_cache: Some(false),
        max_seq_len: Some(64),
        max_batch_size: Some(2),
    };
    let pool = match LayerKVPool::new(cfg, NUM_BLOCKS, cache_dtype) {
        Ok(p) => Arc::new(p),
        Err(e) if e.contains("No Metal device found") => {
            eprintln!("skipping: {e}");
            return None;
        }
        Err(e) if e.contains("Metal GPU not available") => {
            eprintln!("skipping: {e}");
            return None;
        }
        Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
    };
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(NUM_BLOCKS, BLOCK_SIZE)));
    let adapter = PagedKVCacheAdapter::new(Arc::clone(&allocator), Arc::clone(&pool), BLOCK_SIZE)
        .expect("PagedKVCacheAdapter::new must succeed when pool/allocator agree");
    Some(Fixture { adapter, allocator })
}

struct Fixture {
    adapter: PagedKVCacheAdapter,
    allocator: Arc<Mutex<BlockAllocator>>,
}

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------

/// Deterministic small pseudo-random floats in roughly `[-0.5, 0.5]`. Chosen
/// to keep softmax inputs well within fp16's exponent range (so the
/// reference and adapter paths don't drift purely from intermediate
/// overflow). Linear-congruential is fine — we just need a reproducible
/// stream that's not constant.
fn sample_floats(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // High 24 bits as a [-0.5, 0.5) float.
        let bits = (state >> 40) as u32; // 24 bits
        let normalized = (bits as f32) / ((1u32 << 24) as f32) - 0.5;
        out.push(normalized);
    }
    out
}

fn floats_to_f16_bits(src: &[f32]) -> Vec<u16> {
    src.iter().map(|x| f16::from_f32(*x).to_bits()).collect()
}

fn floats_to_bf16_bits(src: &[f32]) -> Vec<u16> {
    src.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

/// Build an MxArray with the requested shape from f32 source data,
/// converting to the target dtype. Float16/BFloat16 use the raw `from_*`
/// constructors so we don't pay an `astype` graph node.
fn array_from_f32(data: &[f32], shape: &[i64], dtype: DType) -> MxArray {
    match dtype {
        DType::Float16 => {
            let bits = floats_to_f16_bits(data);
            MxArray::from_float16(&bits, shape).expect("from_float16")
        }
        DType::BFloat16 => {
            let bits = floats_to_bf16_bits(data);
            MxArray::from_bfloat16(&bits, shape).expect("from_bfloat16")
        }
        DType::Float32 => MxArray::from_float32(data, shape).expect("from_float32"),
        other => panic!("unsupported dtype for test fixtures: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Reference path: flat attention via MLX SDPA
// ---------------------------------------------------------------------------

/// Compute the reference attention output via MLX's `scaled_dot_product_attention`.
///
/// Inputs are laid out per-token (matching the adapter's `update_keys_values`
/// expectations):
/// - `keys_per_token`, `values_per_token`: `[num_tokens, num_kv_heads, head_size]`
/// - `query`: `[1, num_query_heads, head_size]`
///
/// SDPA expects `[batch, n_heads, seq_len, head_dim]`, so we reshape:
/// - Q → `[1, num_query_heads, 1, head_size]`
/// - K → `[1, num_kv_heads, num_tokens, head_size]`
/// - V → `[1, num_kv_heads, num_tokens, head_size]`
///
/// With `num_query_heads == num_kv_heads` this is plain MHA.
///
/// Returns the output as a `Vec<f32>` of length `num_query_heads * head_size`.
fn reference_attention_f32(
    keys_per_token: &MxArray,
    values_per_token: &MxArray,
    query: &MxArray,
    head_size: u32,
) -> Vec<f32> {
    let num_tokens = NUM_TOKENS as i64;
    let num_kv = NUM_KV_HEADS as i64;
    let num_q = NUM_QUERY_HEADS as i64;
    let d = head_size as i64;

    // Reshape K/V from [num_tokens, num_kv_heads, head_size]
    // to [1, num_kv_heads, num_tokens, head_size] (transpose 0<->1, add batch).
    let k = keys_per_token
        .reshape(&[num_tokens, num_kv, d])
        .expect("k reshape")
        .transpose(Some(&[1, 0, 2]))
        .expect("k transpose")
        .reshape(&[1, num_kv, num_tokens, d])
        .expect("k reshape final");
    let v = values_per_token
        .reshape(&[num_tokens, num_kv, d])
        .expect("v reshape")
        .transpose(Some(&[1, 0, 2]))
        .expect("v transpose")
        .reshape(&[1, num_kv, num_tokens, d])
        .expect("v reshape final");

    // Reshape Q from [1, num_query_heads, head_size]
    // to [1, num_query_heads, 1, head_size].
    let q = query.reshape(&[1, num_q, 1, d]).expect("q reshape");

    let scale = (head_size as f64).sqrt().recip();
    let out = scaled_dot_product_attention(&q, &k, &v, scale, None)
        .expect("scaled_dot_product_attention must succeed");

    // Output shape: [1, num_query_heads, 1, head_size]. Materialize and
    // copy to host as f32 for comparison.
    out.eval();
    synchronize();
    let out_f32 = out
        .astype(DType::Float32)
        .expect("astype f32")
        .to_float32()
        .expect("to_float32");
    let arr: &[f32] = out_f32.as_ref();
    arr.to_vec()
}

// ---------------------------------------------------------------------------
// Adapter path: paged attention via the full adapter API
// ---------------------------------------------------------------------------

/// Run the FULL adapter flow: reset, find_cached_prefix (miss), allocate
/// suffix, record_tokens, batch update_keys_values, gather_kv_for_decode.
/// Returns the gather output as `Vec<f32>` for comparison. The adapter's
/// decode output stays in the query/io dtype, so tests cast before copying
/// back to host.
#[allow(clippy::too_many_arguments)]
fn run_adapter(
    fixture: &mut Fixture,
    seq_id: u32,
    token_ids: &[u32],
    keys_per_token: &MxArray,
    values_per_token: &MxArray,
    query: &MxArray,
    head_size: u32,
    register_for_reuse: bool,
) -> Vec<f32> {
    fixture
        .adapter
        .reset_for_new_request(seq_id)
        .expect("reset_for_new_request");

    // Prefix lookup — fresh allocator on the first call so this misses; on
    // the prefix-reuse test we pass a pre-seeded allocator and this hits.
    let cached = fixture
        .adapter
        .find_cached_prefix(token_ids, &[], 0, false)
        .expect("find_cached_prefix");

    // Always allocate enough suffix blocks to cover the full token range
    // (allocate_suffix_blocks computes total_tokens − cached internally).
    fixture
        .adapter
        .allocate_suffix_blocks(token_ids.len() as u32)
        .expect("allocate_suffix_blocks");

    // Record only the suffix tokens (find_cached_prefix already seeded the
    // prefix). On a miss `cached.cached_token_count == 0`, so this records
    // every token; on a hit we record only the new portion (here: nothing,
    // since the test reuses the entire prefix).
    let cached_n = cached.cached_token_count as usize;
    if cached_n < token_ids.len() {
        fixture
            .adapter
            .record_tokens(&token_ids[cached_n..])
            .expect("record_tokens");
    }

    // Write KV for the suffix range only — the prefix's KV is already in
    // the pool's cache from the previous request.
    let suffix_n = token_ids.len() - cached_n;
    if suffix_n > 0 {
        // Build per-suffix slices of K/V. `keys_per_token` holds all
        // num_tokens × num_kv_heads × head_size elements; we slice along
        // axis 0.
        let suffix_keys = keys_per_token
            .slice(
                &[cached_n as i64, 0, 0],
                &[
                    token_ids.len() as i64,
                    NUM_KV_HEADS as i64,
                    head_size as i64,
                ],
            )
            .expect("slice keys");
        let suffix_values = values_per_token
            .slice(
                &[cached_n as i64, 0, 0],
                &[
                    token_ids.len() as i64,
                    NUM_KV_HEADS as i64,
                    head_size as i64,
                ],
            )
            .expect("slice values");
        suffix_keys.eval();
        suffix_values.eval();
        synchronize();
        fixture
            .adapter
            .update_keys_values(
                /* layer_idx */ 0,
                &suffix_keys,
                &suffix_values,
                /* first_logical_position */ cached_n as u32,
            )
            .expect("update_keys_values");
    }

    // Gather attention. scale = 1/sqrt(head_size); softcap = 1.0 disables.
    let scale = (head_size as f32).sqrt().recip();
    let out = fixture
        .adapter
        .gather_kv_for_decode(/* layer_idx */ 0, query, scale, /* softcap */ 1.0)
        .expect("gather_kv_for_decode");
    out.eval();
    synchronize();

    let out_f32 = out
        .astype(DType::Float32)
        .expect("astype f32")
        .to_float32()
        .expect("to_float32");
    let result: Vec<f32> = (out_f32.as_ref() as &[f32]).to_vec();

    if register_for_reuse {
        fixture
            .adapter
            .register_full_blocks_for_reuse(&[], 0)
            .expect("register_full_blocks_for_reuse");
    }
    fixture.adapter.release_request().expect("release_request");

    result
}

// ---------------------------------------------------------------------------
// Comparison helper
// ---------------------------------------------------------------------------

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "output length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Locate the worst-diverging element so a failed assertion produces a
/// useful repro hint instead of "fourth element looks fine, take my word
/// for it".
fn argmax_abs_diff(a: &[f32], b: &[f32]) -> (usize, f32, f32, f32) {
    let mut idx = 0usize;
    let mut best = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > best {
            best = d;
            idx = i;
        }
    }
    (idx, best, a[idx], b[idx])
}

// ---------------------------------------------------------------------------
// Test variants
// ---------------------------------------------------------------------------

/// Build per-token-varying K/V/Q for a token-aware mean-of-V check.
///
/// `V[t, h, d] = (t + 1) + 0.01 * (h + d)`, K = 0, Q = 0. With Q · K = 0
/// the softmax is uniform 1/N, so the expected output is
/// `mean_t(V) = (N+1)/2 + 0.01*(h+d)` per (h, d).
///
/// Token-varying V lets us tell which tokens the kernel actually read
/// (constant-V hides "I only read block 0"-style multi-block bugs).
fn run_zero_q_mean_v(fixture: &mut Fixture, n_tokens: u32, seq_id: u32) -> (Vec<f32>, Vec<f32>) {
    let n = n_tokens as usize;
    let mut k_data = vec![0.0f32; n * NUM_KV_HEADS as usize * HEAD_SIZE as usize];
    let mut v_data = vec![0.0f32; n * NUM_KV_HEADS as usize * HEAD_SIZE as usize];
    for t in 0..n {
        for h in 0..NUM_KV_HEADS as usize {
            for d in 0..HEAD_SIZE as usize {
                let i = t * NUM_KV_HEADS as usize * HEAD_SIZE as usize + h * HEAD_SIZE as usize + d;
                v_data[i] = (t + 1) as f32 + 0.01 * (h + d) as f32;
                k_data[i] = 0.0;
            }
        }
    }
    let q_data = vec![0.0f32; NUM_QUERY_HEADS as usize * HEAD_SIZE as usize];
    let kv_shape = [n_tokens as i64, NUM_KV_HEADS as i64, HEAD_SIZE as i64];
    let q_shape = [1i64, NUM_QUERY_HEADS as i64, HEAD_SIZE as i64];
    let keys = array_from_f32(&k_data, &kv_shape, DType::Float16);
    let values = array_from_f32(&v_data, &kv_shape, DType::Float16);
    let query = array_from_f32(&q_data, &q_shape, DType::Float16);
    keys.eval();
    values.eval();
    query.eval();
    synchronize();
    let token_ids: Vec<u32> = (0..n_tokens).collect();
    let out = run_adapter(
        fixture, seq_id, &token_ids, &keys, &values, &query, HEAD_SIZE, false,
    );
    let mean_t = (n_tokens as f32 + 1.0) / 2.0;
    let expected: Vec<f32> = (0..NUM_QUERY_HEADS as usize)
        .flat_map(|h| (0..HEAD_SIZE as usize).map(move |d| mean_t + 0.01 * (h + d) as f32))
        .collect();
    (out, expected)
}

/// Block-count scaling: with `n_tokens ∈ {8, 16, 24}` (= 1, 2, 3 blocks of
/// size 8) and Q=K=0, the kernel's V output reduction across warps must
/// produce `mean_t(V)` for every output element. A regression in the
/// V-reduction (e.g. truncated threadgroup memory dropping upper-warp
/// contributions) shows up as a large deviation on the trailing
/// `head_size` dimensions for the multi-block runs.
///
/// Discovered during P1C-4 development: the original
/// `dispatch_paged_attention_v1_raw` allocation only sized for the QK
/// softmax phase's `logits[max_seq_len]`, silently truncating the
/// V-reduction's reinterpreted `out_smem` (`(NUM_WARPS/2) * HEAD_SIZE`
/// f32s = 1024 bytes for HEAD_SIZE=64). Production never tripped it
/// because real contexts always satisfied
/// `max_seq_len * 4 ≥ (NUM_WARPS/2) * HEAD_SIZE * 4`.
#[test]
fn test_adapter_e2e_block_count_scaling() {
    let Some(mut fixture) = try_make_fixture(MetalDtype::Float16) else {
        return;
    };
    for &n in &[8u32, 16u32, 24u32] {
        let (got, expected) = run_zero_q_mean_v(&mut fixture, n, 0);
        let diff = max_abs_diff(&got, &expected);
        let (idx, worst, g, e) = argmax_abs_diff(&got, &expected);
        assert!(
            diff < 1e-2,
            "block-count scaling failed for n_tokens={n}: max_abs_diff={diff} (>= 1e-2). \
             Worst idx {idx}: got {g}, exp {e}, |d|={worst}. Indicates a regression in \
             the V-reduction across warps (e.g. threadgroup_mem_size truncation).",
        );
        eprintln!("n_tokens = {n}: max_abs_diff = {diff}");
    }
}

/// Float16 cache + Float16 K/V/Q. Asserts adapter output ≈ flat-MLX SDPA
/// output within fp16 tolerance.
///
/// The 1e-2 tolerance reflects the cumulative effect of:
/// - kernel-side fp16 accumulation (vs MLX SDPA's higher-precision accum),
/// - reorder differences between the paged kernel's per-block reduction and
///   the flat SDPA's per-token reduction.
///
/// Empirically observed runs hit the low 1e-3 range on M3-class hardware.
/// We leave headroom for older silicon and BF16 path drift in the sister
/// test below.
#[test]
fn test_adapter_e2e_matches_flat_attention_fp16() {
    let dtype = DType::Float16;
    let metal_dtype = MetalDtype::Float16;

    let Some(mut fixture) = try_make_fixture(metal_dtype) else {
        return; // skipped: no Metal device.
    };

    // Synthetic per-token K/V and a single query.
    let kv_len = (NUM_TOKENS * NUM_KV_HEADS * HEAD_SIZE) as usize;
    let q_len = (NUM_QUERY_HEADS * HEAD_SIZE) as usize;
    let k_data = sample_floats(0xA1, kv_len);
    let v_data = sample_floats(0xB2, kv_len);
    let q_data = sample_floats(0xC3, q_len);

    let kv_shape = [NUM_TOKENS as i64, NUM_KV_HEADS as i64, HEAD_SIZE as i64];
    let q_shape = [1i64, NUM_QUERY_HEADS as i64, HEAD_SIZE as i64];

    let keys = array_from_f32(&k_data, &kv_shape, dtype);
    let values = array_from_f32(&v_data, &kv_shape, dtype);
    let query = array_from_f32(&q_data, &q_shape, dtype);
    keys.eval();
    values.eval();
    query.eval();
    synchronize();

    let token_ids: Vec<u32> = (0..NUM_TOKENS).collect();

    // Reference: flat-MLX SDPA over the SAME K/V/Q tensors.
    let ref_out = reference_attention_f32(&keys, &values, &query, HEAD_SIZE);

    // Adapter: write K/V via update_keys_values, gather via gather_kv_for_decode.
    let adapter_out = run_adapter(
        &mut fixture,
        /* seq_id */ 0,
        &token_ids,
        &keys,
        &values,
        &query,
        HEAD_SIZE,
        /* register_for_reuse */ false,
    );

    let diff = max_abs_diff(&ref_out, &adapter_out);
    let (idx, worst, r, a) = argmax_abs_diff(&ref_out, &adapter_out);
    assert!(
        diff < 1e-2,
        "FP16 adapter output diverges from flat reference: max_abs_diff = {diff} (>= 1e-2). \
         Worst at idx {idx}: ref = {r}, adapter = {a}, |diff| = {worst}. \
         len = {}",
        ref_out.len(),
    );
    eprintln!("FP16 e2e: max_abs_diff = {diff} (worst at idx {idx})");
}

/// BFloat16 cache + BFloat16 K/V/Q. BF16 has 8-bit mantissa vs FP16's 10-bit,
/// so allow 5e-2 tolerance.
#[test]
fn test_adapter_e2e_matches_flat_attention_bf16() {
    let dtype = DType::BFloat16;
    let metal_dtype = MetalDtype::BFloat16;

    let Some(mut fixture) = try_make_fixture(metal_dtype) else {
        return; // skipped: no Metal device.
    };

    let kv_len = (NUM_TOKENS * NUM_KV_HEADS * HEAD_SIZE) as usize;
    let q_len = (NUM_QUERY_HEADS * HEAD_SIZE) as usize;
    let k_data = sample_floats(0xA1, kv_len);
    let v_data = sample_floats(0xB2, kv_len);
    let q_data = sample_floats(0xC3, q_len);

    let kv_shape = [NUM_TOKENS as i64, NUM_KV_HEADS as i64, HEAD_SIZE as i64];
    let q_shape = [1i64, NUM_QUERY_HEADS as i64, HEAD_SIZE as i64];

    let keys = array_from_f32(&k_data, &kv_shape, dtype);
    let values = array_from_f32(&v_data, &kv_shape, dtype);
    let query = array_from_f32(&q_data, &q_shape, dtype);
    keys.eval();
    values.eval();
    query.eval();
    synchronize();

    let token_ids: Vec<u32> = (0..NUM_TOKENS).collect();

    let ref_out = reference_attention_f32(&keys, &values, &query, HEAD_SIZE);
    let adapter_out = run_adapter(
        &mut fixture,
        0,
        &token_ids,
        &keys,
        &values,
        &query,
        HEAD_SIZE,
        false,
    );

    let diff = max_abs_diff(&ref_out, &adapter_out);
    let (idx, worst, r, a) = argmax_abs_diff(&ref_out, &adapter_out);
    assert!(
        diff < 5e-2,
        "BF16 adapter output diverges from flat reference: max_abs_diff = {diff} (>= 5e-2). \
         Worst at idx {idx}: ref = {r}, adapter = {a}, |diff| = {worst}. \
         len = {}",
        ref_out.len(),
    );
    eprintln!("BF16 e2e: max_abs_diff = {diff} (worst at idx {idx})");
}

/// Prefix-reuse semantics: the SECOND request, served from the prefix
/// cache (no fresh KV write), must produce the same output as the FIRST
/// request that wrote the KV. Validates that `find_cached_prefix` +
/// `register_full_blocks_for_reuse` actually preserve the on-GPU KV
/// contents across requests instead of just re-incrementing refcounts on
/// stale blocks.
#[test]
fn test_adapter_e2e_prefix_reuse_matches_no_reuse() {
    let dtype = DType::Float16;
    let metal_dtype = MetalDtype::Float16;

    let Some(mut fixture) = try_make_fixture(metal_dtype) else {
        return;
    };

    let kv_len = (NUM_TOKENS * NUM_KV_HEADS * HEAD_SIZE) as usize;
    let q_len = (NUM_QUERY_HEADS * HEAD_SIZE) as usize;
    let k_data = sample_floats(0xA1, kv_len);
    let v_data = sample_floats(0xB2, kv_len);
    let q_data = sample_floats(0xC3, q_len);

    let kv_shape = [NUM_TOKENS as i64, NUM_KV_HEADS as i64, HEAD_SIZE as i64];
    let q_shape = [1i64, NUM_QUERY_HEADS as i64, HEAD_SIZE as i64];

    let keys = array_from_f32(&k_data, &kv_shape, dtype);
    let values = array_from_f32(&v_data, &kv_shape, dtype);
    let query = array_from_f32(&q_data, &q_shape, dtype);
    keys.eval();
    values.eval();
    query.eval();
    synchronize();

    // Use exactly 24 tokens = 3 full blocks of size 8. All of them are
    // eligible for prefix-cache registration.
    let token_ids: Vec<u32> = (0..NUM_TOKENS).collect();

    // Request 1: writes KV, gathers, then registers for reuse.
    let initial_free = fixture.allocator.lock().unwrap().num_free_blocks();
    let first_out = run_adapter(
        &mut fixture,
        0,
        &token_ids,
        &keys,
        &values,
        &query,
        HEAD_SIZE,
        /* register_for_reuse */ true,
    );

    // Confirm registration actually pinned blocks in the prefix cache.
    let after_first = fixture.allocator.lock().unwrap().num_free_blocks();
    assert!(
        after_first < initial_free,
        "register_full_blocks_for_reuse must pin blocks in the prefix cache \
         (initial_free = {initial_free}, after = {after_first})"
    );

    // Request 2: same token_ids → prefix cache hits ALL three blocks.
    // No update_keys_values call for the cached prefix (run_adapter skips
    // when cached_n covers all tokens). gather_kv_for_decode reads the
    // SAME GPU buffers Request 1 wrote. Output must match bit-for-bit
    // modulo nondeterministic kernel reduction order.
    fixture
        .adapter
        .reset_for_new_request(1)
        .expect("reset_for_new_request");
    let cached = fixture
        .adapter
        .find_cached_prefix(&token_ids, &[], 0, false)
        .expect("find_cached_prefix on second request must succeed");
    assert_eq!(
        cached.cached_token_count, NUM_TOKENS,
        "second request must hit the full {NUM_TOKENS}-token prefix"
    );
    assert_eq!(
        cached.blocks.len(),
        (NUM_TOKENS / BLOCK_SIZE) as usize,
        "expected {} prefix blocks reused",
        NUM_TOKENS / BLOCK_SIZE
    );

    // No suffix to allocate (total_tokens == cached_token_count).
    let n_alloc = fixture
        .adapter
        .allocate_suffix_blocks(NUM_TOKENS)
        .expect("allocate_suffix_blocks");
    assert_eq!(
        n_alloc, 0,
        "second request reuses the entire prefix; zero new blocks expected"
    );

    // Skip update_keys_values entirely — the prefix's KV is already in
    // place. Gather directly.
    let scale = (HEAD_SIZE as f32).sqrt().recip();
    let out2 = fixture
        .adapter
        .gather_kv_for_decode(0, &query, scale, 1.0)
        .expect("gather_kv_for_decode on reused prefix");
    out2.eval();
    synchronize();
    let out2_f32 = out2.to_float32().expect("to_float32");
    let second_out: Vec<f32> = (out2_f32.as_ref() as &[f32]).to_vec();
    fixture.adapter.release_request().expect("release_request");

    let diff = max_abs_diff(&first_out, &second_out);
    // Tighter tolerance than the absolute fp16 test: the SAME kernel runs
    // against the SAME on-GPU buffers, so any drift would indicate either
    // (a) a non-deterministic reduction order on the gather kernel, or
    // (b) a stale-KV bug in the prefix-reuse path. 1e-3 catches both.
    assert!(
        diff < 1e-3,
        "Prefix-reuse output diverges from no-reuse output: max_abs_diff = {diff} (>= 1e-3). \
         No-reuse[..4] = {:?}, Reuse[..4] = {:?}",
        &first_out[..first_out.len().min(4)],
        &second_out[..second_out.len().min(4)],
    );
    eprintln!("FP16 prefix-reuse: max_abs_diff = {diff}");
}
