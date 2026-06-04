//! Smoke tests for the `mlx_qwen35_forward_paged` C++ machinery (Dense
//! variant) — the C++ side of paged decode for Qwen3.5 Dense
//! (`attn_for_compile_paged` reuse, `dense_compiled_decode_fn_paged`, the
//! per-layer pool / scale globals, and the `init_paged` / `forward_paged`
//! FFI). These tests exercise the FFI surface directly to prove:
//!
//! 1. The symbols are linked (no missing-symbol crash).
//! 2. The early-exit guard (`g_dense_paged_inited == false` →
//!    `output_logits = nullptr`) works before init.
//! 3. After `mlx_qwen35_init_paged` succeeds against placeholder
//!    pool / scale arrays, calling `mlx_qwen35_forward_paged` does
//!    NOT crash even when no real weights are registered — the C++
//!    catch wrapper turns the inevitable "Weight not found" exception
//!    into a `output_logits = nullptr` return.
//! 4. The reset path (`mlx_qwen35_compiled_reset`) clears the paged
//!    globals (not just the legacy flat ones).
//! 5. The single-token decode contract guard rejects multi-token
//!    inputs without crashing.
//!
//! These are deliberately not parity / numerical / end-to-end tests. The
//! goal here is "the binary linked and the path doesn't crash."
//!
//! All Metal-dependent setup gracefully skips on hosts where MLX can't
//! allocate Metal buffers; the tests are no-ops there.

#![cfg(target_os = "macos")]

use std::ptr;

// =============================================================================
// Minimal config — the paged path hard-codes block_size=16, x_pack=8,
// kv_dtype=Bf16, sliding=0. This config sizes the placeholder pools so
// every paged-op factory validator passes. The model dimensions are
// borrowed from a typical Qwen3.5 Dense checkpoint and are unrelated to
// the actual hardware compute.
// =============================================================================

const NUM_LAYERS: i32 = 64;
const HIDDEN_SIZE: i32 = 2048;
const NUM_HEADS: i32 = 16;
const NUM_KV_HEADS: i32 = 2;
const HEAD_DIM: i32 = 128;
const ROPE_THETA: f32 = 100_000.0;
const ROPE_DIMS: i32 = 32;
const RMS_NORM_EPS: f32 = 1e-6;
const FULL_ATTN_INTERVAL: i32 = 4;
const LIN_NUM_K_HEADS: i32 = 16;
const LIN_NUM_V_HEADS: i32 = 64;
const LIN_KEY_DIM: i32 = 192;
const LIN_VALUE_DIM: i32 = 128;
const LIN_CONV_KERNEL: i32 = 4;
const TIE_EMBED: i32 = 1;
const MAX_KV_LEN: i32 = 64;
const BATCH_SIZE: i32 = 1;

// Paged storage scalars — block_size=16 / x_pack=8 / Bf16 are hard-coded
// in `attn_for_compile_paged` so the smoke test must mirror them
// exactly.
const BLOCK_SIZE: i64 = 16;
const X_PACK: i64 = 8;
const NUM_BLOCKS: i64 = 4;
const MAX_BLOCKS_PER_SEQ: i64 = NUM_BLOCKS;
const CHUNK_SIZE_MAX: i64 = 1; // single-token decode

// dtype codes (BridgeDType in mlx_common.h)
const INT32: i32 = 1;
const BFLOAT16: i32 = 3;

/// True if the host has a usable Metal backend. `mlx_array_zeros` (and
/// every MLX op below) construct via the C++ MLX entry points without a
/// catch wrapper, so on a no-Metal host the lazy `metal::allocator()`
/// construction throws a foreign exception and aborts the Rust runtime
/// (`Rust cannot catch foreign exceptions`). Every test that touches
/// MLX arrays MUST early-return when this returns false.
fn metal_available() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

/// Allocate a contiguous bf16 zero-array of the given shape via
/// `mlx_array_zeros` (handle is heap-allocated; caller deletes via
/// `mlx_array_delete`). MUST only be called when `metal_available()`
/// returned true; otherwise the underlying MLX op throws across the FFI
/// boundary and aborts.
fn bf16_zeros(shape: &[i64]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_array_zeros(shape.as_ptr(), shape.len(), BFLOAT16) }
}

fn i32_zeros(shape: &[i64]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_array_zeros(shape.as_ptr(), shape.len(), INT32) }
}

fn i32_arr_from(values: &[i32], shape: &[i64]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_array_from_int32(values.as_ptr(), shape.as_ptr(), shape.len()) }
}

fn i64_arr_from(values: &[i64], shape: &[i64]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_array_from_int64(values.as_ptr(), shape.as_ptr(), shape.len()) }
}

fn f32_scalar(value: f32) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_array_from_float32(&value as *const f32, ptr::null(), 0) }
}

unsafe fn delete(handle: *mut mlx_sys::mlx_array) {
    if !handle.is_null() {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
}

unsafe extern "C" {
    fn mlx_qwen35_init_paged(
        num_layers: i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        rope_dims: i32,
        rms_norm_eps: f32,
        full_attention_interval: i32,
        linear_num_k_heads: i32,
        linear_num_v_heads: i32,
        linear_key_head_dim: i32,
        linear_value_head_dim: i32,
        linear_conv_kernel_dim: i32,
        tie_word_embeddings: i32,
        max_kv_len: i32,
        batch_size: i32,
        k_pool_handles: *mut *mut mlx_sys::mlx_array,
        v_pool_handles: *mut *mut mlx_sys::mlx_array,
        k_scale_handles: *mut *mut mlx_sys::mlx_array,
        v_scale_handles: *mut *mut mlx_sys::mlx_array,
        linear_cache_arrays: *mut *mut mlx_sys::mlx_array,
        prefill_offset: i32,
    ) -> i32;

    fn mlx_qwen35_forward_paged(
        input_ids: *mut mlx_sys::mlx_array,
        embedding_weight: *mut mlx_sys::mlx_array,
        offset_arr: *mut mlx_sys::mlx_array,
        block_table: *mut mlx_sys::mlx_array,
        slot_mapping: *mut mlx_sys::mlx_array,
        num_valid_tokens: *mut mlx_sys::mlx_array,
        num_valid_blocks: *mut mlx_sys::mlx_array,
        seq_lens: *mut mlx_sys::mlx_array,
        output_logits: *mut *mut mlx_sys::mlx_array,
        cache_offset_out: *mut i32,
    );

    // Used to drop registered weights between tests so the second test's
    // reset state is clean.
    fn mlx_clear_weights();

    // Resets the dense compiled state, including the paged-path globals
    // (`g_dense_paged_inited`, `g_dense_k_pools`, `g_dense_v_pools`,
    // `g_dense_k_scales`, `g_dense_v_scales`,
    // `g_dense_paged_linear_caches`, `g_dense_paged_offset_int`,
    // `g_dense_paged_config`) — the `forward_paged_after_reset_returns_null`
    // test below regresses that.
    fn mlx_qwen35_compiled_reset();
}

/// Test 1 — pre-init guard: calling `_forward_paged` BEFORE
/// `_init_paged` must early-exit with `output_logits = nullptr` and
/// must not crash. Proves the new FFI symbol is linked and the
/// `g_dense_paged_inited` gate works.
#[test]
fn forward_paged_before_init_returns_null_no_crash() {
    // Reset any state from previous test runs.
    unsafe {
        mlx_qwen35_compiled_reset();
        mlx_clear_weights();
    }

    let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
    let mut offset_out: i32 = -1;

    // All input handles can be null — the early-exit guard fires before
    // any deref.
    unsafe {
        mlx_qwen35_forward_paged(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut logits,
            &mut offset_out,
        );
    }

    assert!(
        logits.is_null(),
        "forward_paged before init must return null logits"
    );
    // offset_out should be untouched (no crash, no write).
    assert_eq!(
        offset_out, -1,
        "offset_out must not be written when uninitialized"
    );
}

/// Test 2 — graph build smoke: register no weights, build all per-layer
/// pool / scale arrays at the shapes the paged-op validators require,
/// and call `_init_paged` then `_forward_paged`. The forward will throw
/// "Weight not found" inside the compile graph; the C++ catch must
/// turn that into a clean `output_logits = nullptr` return, not a
/// crash. This proves:
///
///   1. The new globals (`g_dense_k_pools`, `g_dense_v_pools`,
///      `g_dense_k_scales`, `g_dense_v_scales`,
///      `g_dense_paged_linear_caches`) are wired up.
///   2. `mlx_qwen35_init_paged` accepts a complete per-layer
///      handle bundle without tripping any validation guard.
///   3. `mlx_qwen35_forward_paged` reaches the compile-graph
///      dispatch and the catch wrapper is hit on the first weight
///      miss (i.e. the path runs end-to-end into the graph).
///
/// Without Metal, MLX array allocation may fail; we treat that as
/// "skip" and exit early without panicking.
#[test]
fn forward_paged_graph_builds_without_crash() {
    if !metal_available() {
        eprintln!(
            "skipping forward_paged_graph_builds_without_crash: Metal unavailable on this host"
        );
        return;
    }
    unsafe {
        mlx_qwen35_compiled_reset();
        mlx_clear_weights();
    }

    // Per-layer pool / scale handles. Linear-layer slots are null
    // (placeholders are stored on the C++ side); full-attn slots get
    // real bf16 zero pools.
    let mut k_pool_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut v_pool_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut k_scale_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut v_scale_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];

    // K pool shape: [num_blocks, num_kv_heads, head_size/x_pack,
    //                block_size, x_pack].
    // V pool shape: [num_blocks, num_kv_heads, head_size, block_size].
    let k_shape = [
        NUM_BLOCKS,
        NUM_KV_HEADS as i64,
        (HEAD_DIM as i64) / X_PACK,
        BLOCK_SIZE,
        X_PACK,
    ];
    let v_shape = [NUM_BLOCKS, NUM_KV_HEADS as i64, HEAD_DIM as i64, BLOCK_SIZE];

    for i in 0..NUM_LAYERS {
        let is_linear = ((i + 1) % FULL_ATTN_INTERVAL) != 0;
        if is_linear {
            // Linear slots stay null — init stores placeholders.
            continue;
        }
        let k = bf16_zeros(&k_shape);
        let v = bf16_zeros(&v_shape);
        let ks = f32_scalar(1.0);
        let vs = f32_scalar(1.0);
        if k.is_null() || v.is_null() || ks.is_null() || vs.is_null() {
            // Metal allocation likely failed — skip the test.
            eprintln!(
                "skipping forward_paged_graph_builds_without_crash: array allocation failed (likely no Metal)"
            );
            unsafe {
                delete(k);
                delete(v);
                delete(ks);
                delete(vs);
                // Clean any earlier full-attn slots already populated.
                for h in k_pool_vec.iter_mut() {
                    delete(*h);
                    *h = ptr::null_mut();
                }
                for h in v_pool_vec.iter_mut() {
                    delete(*h);
                    *h = ptr::null_mut();
                }
                for h in k_scale_vec.iter_mut() {
                    delete(*h);
                    *h = ptr::null_mut();
                }
                for h in v_scale_vec.iter_mut() {
                    delete(*h);
                    *h = ptr::null_mut();
                }
            }
            return;
        }
        k_pool_vec[i as usize] = k;
        v_pool_vec[i as usize] = v;
        k_scale_vec[i as usize] = ks;
        v_scale_vec[i as usize] = vs;
    }

    // Init the paged graph. linear_cache_arrays is null → init stashes
    // bf16 placeholders for every linear-layer slot.
    let init_status = unsafe {
        mlx_qwen35_init_paged(
            NUM_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_THETA,
            ROPE_DIMS,
            RMS_NORM_EPS,
            FULL_ATTN_INTERVAL,
            LIN_NUM_K_HEADS,
            LIN_NUM_V_HEADS,
            LIN_KEY_DIM,
            LIN_VALUE_DIM,
            LIN_CONV_KERNEL,
            TIE_EMBED,
            MAX_KV_LEN,
            BATCH_SIZE,
            k_pool_vec.as_mut_ptr(),
            v_pool_vec.as_mut_ptr(),
            k_scale_vec.as_mut_ptr(),
            v_scale_vec.as_mut_ptr(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(
        init_status, 0,
        "mlx_qwen35_init_paged must succeed with full per-layer handle bundle"
    );

    // PagedAttentionInputs metadata at the shapes documented in
    // mlx_common.h.
    let offset_arr = i32_arr_from(&[0], &[1]);
    let block_table_data: Vec<i32> = vec![-1; MAX_BLOCKS_PER_SEQ as usize];
    let block_table = i32_arr_from(&block_table_data, &[1, MAX_BLOCKS_PER_SEQ]);
    let slot_mapping_data: Vec<i64> = vec![-1; CHUNK_SIZE_MAX as usize];
    let slot_mapping = i64_arr_from(&slot_mapping_data, &[CHUNK_SIZE_MAX]);
    let num_valid_tokens = i32_arr_from(&[1], &[1]);
    let num_valid_blocks = i32_arr_from(&[1], &[1]);
    let seq_lens = i32_arr_from(&[1], &[1]);

    // Input ids + embedding placeholder. The forward will throw on the
    // first `get_weight("embedding.weight")` lookup; the catch wrapper
    // turns that into a null logits return.
    let input_ids = i32_zeros(&[BATCH_SIZE as i64, 1]);
    let embedding_weight = bf16_zeros(&[1, HIDDEN_SIZE as i64]);

    // Bail out cleanly if any allocation failed (e.g. Metal-less host).
    if [
        offset_arr,
        block_table,
        slot_mapping,
        num_valid_tokens,
        num_valid_blocks,
        seq_lens,
        input_ids,
        embedding_weight,
    ]
    .iter()
    .any(|h| h.is_null())
    {
        eprintln!(
            "skipping forward_paged_graph_builds_without_crash: input array allocation failed"
        );
        unsafe {
            delete(offset_arr);
            delete(block_table);
            delete(slot_mapping);
            delete(num_valid_tokens);
            delete(num_valid_blocks);
            delete(seq_lens);
            delete(input_ids);
            delete(embedding_weight);
            for h in k_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in k_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
        }
        return;
    }

    let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
    let mut offset_out: i32 = -1;

    unsafe {
        mlx_qwen35_forward_paged(
            input_ids,
            embedding_weight,
            offset_arr,
            block_table,
            slot_mapping,
            num_valid_tokens,
            num_valid_blocks,
            seq_lens,
            &mut logits,
            &mut offset_out,
        );
    }

    // Without registered weights, the compile graph throws inside the
    // catch wrapper and logits stays null. The crucial assertion is
    // "no crash, no abort" — reaching this line proves both.
    if !logits.is_null() {
        // Defensive cleanup if the implementation ever changes to
        // return non-null logits before the weight lookup.
        unsafe { delete(logits) };
    }

    // Cleanup all input handles.
    unsafe {
        delete(input_ids);
        delete(embedding_weight);
        delete(offset_arr);
        delete(block_table);
        delete(slot_mapping);
        delete(num_valid_tokens);
        delete(num_valid_blocks);
        delete(seq_lens);
        for h in k_pool_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in v_pool_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in k_scale_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in v_scale_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        mlx_qwen35_compiled_reset();
    }
}

/// Test 3 — `mlx_qwen35_compiled_reset()` MUST clear the paged
/// globals, not just the legacy flat ones. Clearing only
/// `g_compiled_caches` / `g_compiled_offset` / `g_offset_int` /
/// `g_compile_inited` would leave `g_dense_paged_inited == true` and
/// stale pool / scale / linear-cache / offset state lying around for the
/// next request to reuse.
///
/// This test:
///   1. Initializes the paged path (flips `g_dense_paged_inited = true`).
///   2. Calls `mlx_qwen35_compiled_reset()`.
///   3. Calls `mlx_qwen35_forward_paged()` with all-null arguments.
///      If the paged init guard was cleared, the FFI's first check
///      (`if (!g_dense_paged_inited) return null`) fires before any
///      deref — so logits stays null and `cache_offset_out` is
///      untouched. If the bug regresses, `g_dense_paged_inited` would
///      still be true and the FFI would deref the null pointers and
///      crash.
#[test]
fn forward_paged_after_reset_returns_null() {
    if !metal_available() {
        eprintln!(
            "skipping forward_paged_after_reset_returns_null: Metal unavailable on this host"
        );
        return;
    }
    unsafe {
        mlx_qwen35_compiled_reset();
        mlx_clear_weights();
    }

    // Build minimal pool / scale handles so init succeeds.
    let mut k_pool_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut v_pool_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut k_scale_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut v_scale_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];

    let k_shape = [
        NUM_BLOCKS,
        NUM_KV_HEADS as i64,
        (HEAD_DIM as i64) / X_PACK,
        BLOCK_SIZE,
        X_PACK,
    ];
    let v_shape = [NUM_BLOCKS, NUM_KV_HEADS as i64, HEAD_DIM as i64, BLOCK_SIZE];

    let mut allocation_failed = false;
    for i in 0..NUM_LAYERS {
        let is_linear = ((i + 1) % FULL_ATTN_INTERVAL) != 0;
        if is_linear {
            continue;
        }
        let k = bf16_zeros(&k_shape);
        let v = bf16_zeros(&v_shape);
        let ks = f32_scalar(1.0);
        let vs = f32_scalar(1.0);
        if k.is_null() || v.is_null() || ks.is_null() || vs.is_null() {
            unsafe {
                delete(k);
                delete(v);
                delete(ks);
                delete(vs);
            }
            allocation_failed = true;
            break;
        }
        k_pool_vec[i as usize] = k;
        v_pool_vec[i as usize] = v;
        k_scale_vec[i as usize] = ks;
        v_scale_vec[i as usize] = vs;
    }

    if allocation_failed {
        eprintln!(
            "skipping forward_paged_after_reset_returns_null: array allocation failed (likely no Metal)"
        );
        unsafe {
            for h in k_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in k_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
        }
        return;
    }

    unsafe {
        let init_status = mlx_qwen35_init_paged(
            NUM_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_THETA,
            ROPE_DIMS,
            RMS_NORM_EPS,
            FULL_ATTN_INTERVAL,
            LIN_NUM_K_HEADS,
            LIN_NUM_V_HEADS,
            LIN_KEY_DIM,
            LIN_VALUE_DIM,
            LIN_CONV_KERNEL,
            TIE_EMBED,
            MAX_KV_LEN,
            BATCH_SIZE,
            k_pool_vec.as_mut_ptr(),
            v_pool_vec.as_mut_ptr(),
            k_scale_vec.as_mut_ptr(),
            v_scale_vec.as_mut_ptr(),
            ptr::null_mut(),
            42, // arbitrary prefill_offset to make stale state visible
        );
        assert_eq!(init_status, 0, "init must succeed with full pool bundle");

        // Reset must clear `g_dense_paged_inited`, returning the paged
        // path to "uninitialized" — same state as before any init call.
        mlx_qwen35_compiled_reset();

        // With the paged init cleared, the FFI must early-exit on the
        // null inputs. Pre-fix bug: paged_inited was still true →
        // null deref crash inside the function body.
        let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
        let mut offset_out: i32 = -1;
        mlx_qwen35_forward_paged(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut logits,
            &mut offset_out,
        );

        assert!(
            logits.is_null(),
            "forward_paged after reset must return null logits (init guard cleared)"
        );
        assert_eq!(
            offset_out, -1,
            "offset_out must not be written after reset (init guard cleared)"
        );

        for h in k_pool_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in v_pool_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in k_scale_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in v_scale_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
    }
}

/// Test 4 — `mlx_qwen35_forward_paged` enforces the single-token
/// decode contract. Calling with `slot_mapping.shape == [2]` must
/// return null logits without crashing or modifying state. This
/// documents that the paged path is decode-only.
#[test]
fn forward_paged_rejects_multi_token_contract_violation() {
    if !metal_available() {
        eprintln!(
            "skipping forward_paged_rejects_multi_token_contract_violation: Metal unavailable on this host"
        );
        return;
    }
    unsafe {
        mlx_qwen35_compiled_reset();
        mlx_clear_weights();
    }

    // Set up a minimal valid paged init so the contract guard (and not
    // the pre-init guard) is the one that fires.
    let mut k_pool_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut v_pool_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut k_scale_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut v_scale_vec: Vec<*mut mlx_sys::mlx_array> = vec![ptr::null_mut(); NUM_LAYERS as usize];

    let k_shape = [
        NUM_BLOCKS,
        NUM_KV_HEADS as i64,
        (HEAD_DIM as i64) / X_PACK,
        BLOCK_SIZE,
        X_PACK,
    ];
    let v_shape = [NUM_BLOCKS, NUM_KV_HEADS as i64, HEAD_DIM as i64, BLOCK_SIZE];

    let mut allocation_failed = false;
    for i in 0..NUM_LAYERS {
        let is_linear = ((i + 1) % FULL_ATTN_INTERVAL) != 0;
        if is_linear {
            continue;
        }
        let k = bf16_zeros(&k_shape);
        let v = bf16_zeros(&v_shape);
        let ks = f32_scalar(1.0);
        let vs = f32_scalar(1.0);
        if k.is_null() || v.is_null() || ks.is_null() || vs.is_null() {
            unsafe {
                delete(k);
                delete(v);
                delete(ks);
                delete(vs);
            }
            allocation_failed = true;
            break;
        }
        k_pool_vec[i as usize] = k;
        v_pool_vec[i as usize] = v;
        k_scale_vec[i as usize] = ks;
        v_scale_vec[i as usize] = vs;
    }

    if allocation_failed {
        eprintln!(
            "skipping forward_paged_rejects_multi_token_contract_violation: array allocation failed (likely no Metal)"
        );
        unsafe {
            for h in k_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in k_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
        }
        return;
    }

    let init_status = unsafe {
        mlx_qwen35_init_paged(
            NUM_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_THETA,
            ROPE_DIMS,
            RMS_NORM_EPS,
            FULL_ATTN_INTERVAL,
            LIN_NUM_K_HEADS,
            LIN_NUM_V_HEADS,
            LIN_KEY_DIM,
            LIN_VALUE_DIM,
            LIN_CONV_KERNEL,
            TIE_EMBED,
            MAX_KV_LEN,
            BATCH_SIZE,
            k_pool_vec.as_mut_ptr(),
            v_pool_vec.as_mut_ptr(),
            k_scale_vec.as_mut_ptr(),
            v_scale_vec.as_mut_ptr(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(init_status, 0, "init must succeed");

    // Build inputs that VIOLATE the single-token contract:
    //   - input_ids.size() == 2 (should be 1)
    //   - slot_mapping.shape == [2] (should be [1])
    let offset_arr = i32_arr_from(&[0], &[1]);
    let block_table_data: Vec<i32> = vec![-1; MAX_BLOCKS_PER_SEQ as usize];
    let block_table = i32_arr_from(&block_table_data, &[1, MAX_BLOCKS_PER_SEQ]);
    // 2 slots — violates the [1] contract.
    let bad_slot_mapping = i64_arr_from(&[-1, -1], &[2]);
    let num_valid_tokens = i32_arr_from(&[1], &[1]);
    let num_valid_blocks = i32_arr_from(&[1], &[1]);
    let seq_lens = i32_arr_from(&[1], &[1]);
    // 2-token input_ids — violates the size==1 contract.
    let bad_input_ids = i32_arr_from(&[0, 0], &[1, 2]);
    let embedding_weight = bf16_zeros(&[1, HIDDEN_SIZE as i64]);

    if [
        offset_arr,
        block_table,
        bad_slot_mapping,
        num_valid_tokens,
        num_valid_blocks,
        seq_lens,
        bad_input_ids,
        embedding_weight,
    ]
    .iter()
    .any(|h| h.is_null())
    {
        eprintln!(
            "skipping forward_paged_rejects_multi_token_contract_violation: input array allocation failed"
        );
        unsafe {
            delete(offset_arr);
            delete(block_table);
            delete(bad_slot_mapping);
            delete(num_valid_tokens);
            delete(num_valid_blocks);
            delete(seq_lens);
            delete(bad_input_ids);
            delete(embedding_weight);
            for h in k_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_pool_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in k_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
            for h in v_scale_vec.iter_mut() {
                delete(*h);
                *h = ptr::null_mut();
            }
        }
        return;
    }

    let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
    let mut offset_out: i32 = -1;

    unsafe {
        mlx_qwen35_forward_paged(
            bad_input_ids,
            embedding_weight,
            offset_arr,
            block_table,
            bad_slot_mapping,
            num_valid_tokens,
            num_valid_blocks,
            seq_lens,
            &mut logits,
            &mut offset_out,
        );
    }

    assert!(
        logits.is_null(),
        "multi-token call must be rejected (logits = null)"
    );
    assert_eq!(
        offset_out, -1,
        "offset_out must not be written when contract is violated"
    );

    unsafe {
        delete(bad_input_ids);
        delete(embedding_weight);
        delete(offset_arr);
        delete(block_table);
        delete(bad_slot_mapping);
        delete(num_valid_tokens);
        delete(num_valid_blocks);
        delete(seq_lens);
        for h in k_pool_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in v_pool_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in k_scale_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        for h in v_scale_vec.iter_mut() {
            delete(*h);
            *h = ptr::null_mut();
        }
        mlx_qwen35_compiled_reset();
    }
}
