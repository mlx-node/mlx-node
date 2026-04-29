//! Phase 7 smoke tests for the new `mlx_gemma4_forward_paged` C++
//! machinery (Gemma4 26B-a4b paged variant).
//!
//! Phase 7 lands the C++ side of paged decode for Gemma4
//! (`gemma4_attention_paged`, `gemma4_compiled_decode_fn_paged`, the
//! per-layer pool / scale globals, and the new `init_paged` /
//! `forward_paged` FFI). The Rust dispatcher integration (chat
//! sync/stream wiring) is reserved for a follow-up. These smoke tests
//! exercise the FFI surface directly to prove:
//!
//! 1. The new symbols are linked (no missing-symbol crash).
//! 2. The early-exit guard (`g_gemma4_paged_inited == false` →
//!    `output_logits = nullptr`) works before init.
//! 3. After `mlx_gemma4_init_paged` succeeds against placeholder
//!    pool / scale arrays, calling `mlx_gemma4_forward_paged` does
//!    NOT crash even when no real weights are registered — the C++
//!    catch wrapper turns the inevitable "Weight not found" exception
//!    into a `output_logits = nullptr` return.
//! 4. The reset path (`mlx_gemma4_reset`) clears the paged globals.
//! 5. The single-token decode contract guard rejects multi-token
//!    inputs without crashing.
//!
//! All Metal-dependent setup gracefully skips on hosts where MLX
//! can't allocate Metal buffers.

#![cfg(target_os = "macos")]

use std::ptr;

// =============================================================================
// Minimal Gemma4 26B-a4b config — chosen so the placeholder pool
// shapes pass the per-layer paged-op factory validators without
// needing real weights. Sliding layers (4 of 5) and a single global
// layer (the 5th) cover both code paths in `gemma4_attention_paged`.
// =============================================================================

const NUM_LAYERS: i32 = 5;
const HIDDEN_SIZE: i32 = 256;
const NUM_HEADS: i32 = 4;
const NUM_KV_HEADS: i32 = 1; // sliding
const HEAD_DIM: i32 = 64; // sliding
const GLOBAL_NUM_KV_HEADS: i32 = 1;
const GLOBAL_HEAD_DIM: i32 = 64; // keep the same so the same pool shape works for global too
const ROPE_THETA: f32 = 1_000_000.0;
const ROPE_LOCAL_BASE_FREQ: f32 = 10_000.0;
const PARTIAL_ROTARY_FACTOR: f32 = 0.25;
const RMS_NORM_EPS: f32 = 1e-6;
const SLIDING_WINDOW: i32 = 32;
const TIE_EMBED: i32 = 1;
const MAX_KV_LEN: i32 = 64;
const BATCH_SIZE: i32 = 1;
const NUM_EXPERTS: i32 = 0; // disable MoE for the smoke test
const TOP_K_EXPERTS: i32 = 0;
const MOE_INTERMEDIATE_SIZE: i32 = 0;
const INTERMEDIATE_SIZE: i32 = 512;
const FINAL_LOGIT_SOFTCAPPING: f32 = 30.0;

// Per-layer types: layers 0..4 alternate sliding (0) / global (1) the way
// real Gemma4 does. We pick layer 4 as the only global one for this
// minimal test.
const LAYER_TYPES: [i32; NUM_LAYERS as usize] = [0, 0, 0, 0, 1];

// Paged storage scalars.
const BLOCK_SIZE: i64 = 16;
const X_PACK: i64 = 8;
const NUM_BLOCKS: i64 = 4;
const MAX_BLOCKS_PER_SEQ: i64 = NUM_BLOCKS;
const CHUNK_SIZE_MAX: i64 = 1; // single-token decode

const INT32: i32 = 1;
const BFLOAT16: i32 = 3;

fn metal_available() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

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

/// Build per-layer pool / scale handles. Layer pools are sized
/// `[NUM_BLOCKS, kv_heads, head_dim/x_pack, BLOCK_SIZE, x_pack]`
/// where (kv_heads, head_dim) match each layer's type.
fn build_layer_handles() -> Option<(
    Vec<*mut mlx_sys::mlx_array>,
    Vec<*mut mlx_sys::mlx_array>,
    Vec<*mut mlx_sys::mlx_array>,
    Vec<*mut mlx_sys::mlx_array>,
)> {
    let mut k_vec = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut v_vec = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut ks_vec = vec![ptr::null_mut(); NUM_LAYERS as usize];
    let mut vs_vec = vec![ptr::null_mut(); NUM_LAYERS as usize];

    for i in 0..(NUM_LAYERS as usize) {
        let is_global = LAYER_TYPES[i] == 1;
        let kv_heads = if is_global {
            GLOBAL_NUM_KV_HEADS
        } else {
            NUM_KV_HEADS
        } as i64;
        let hd = if is_global { GLOBAL_HEAD_DIM } else { HEAD_DIM } as i64;
        let k_shape = [NUM_BLOCKS, kv_heads, hd / X_PACK, BLOCK_SIZE, X_PACK];
        let v_shape = [NUM_BLOCKS, kv_heads, hd, BLOCK_SIZE];

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
            // Roll back any earlier slots and bail.
            for h in k_vec.iter_mut() {
                unsafe { delete(*h) };
                *h = ptr::null_mut();
            }
            for h in v_vec.iter_mut() {
                unsafe { delete(*h) };
                *h = ptr::null_mut();
            }
            for h in ks_vec.iter_mut() {
                unsafe { delete(*h) };
                *h = ptr::null_mut();
            }
            for h in vs_vec.iter_mut() {
                unsafe { delete(*h) };
                *h = ptr::null_mut();
            }
            return None;
        }
        k_vec[i] = k;
        v_vec[i] = v;
        ks_vec[i] = ks;
        vs_vec[i] = vs;
    }
    Some((k_vec, v_vec, ks_vec, vs_vec))
}

unsafe fn release_layer_handles(
    k_vec: &mut [*mut mlx_sys::mlx_array],
    v_vec: &mut [*mut mlx_sys::mlx_array],
    ks_vec: &mut [*mut mlx_sys::mlx_array],
    vs_vec: &mut [*mut mlx_sys::mlx_array],
) {
    for h in k_vec.iter_mut() {
        unsafe { delete(*h) };
        *h = ptr::null_mut();
    }
    for h in v_vec.iter_mut() {
        unsafe { delete(*h) };
        *h = ptr::null_mut();
    }
    for h in ks_vec.iter_mut() {
        unsafe { delete(*h) };
        *h = ptr::null_mut();
    }
    for h in vs_vec.iter_mut() {
        unsafe { delete(*h) };
        *h = ptr::null_mut();
    }
}

unsafe extern "C" {
    fn mlx_gemma4_init_paged(
        num_layers: i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        global_num_kv_heads: i32,
        global_head_dim: i32,
        rope_theta: f32,
        rope_local_base_freq: f32,
        partial_rotary_factor: f32,
        rms_norm_eps: f32,
        sliding_window: i32,
        tie_word_embeddings: i32,
        max_kv_len: i32,
        batch_size: i32,
        num_experts: i32,
        top_k_experts: i32,
        moe_intermediate_size: i32,
        intermediate_size: i32,
        final_logit_softcapping: f32,
        layer_types: *const i32,
        layer_types_len: i32,
        k_pool_handles: *mut *mut mlx_sys::mlx_array,
        v_pool_handles: *mut *mut mlx_sys::mlx_array,
        k_scale_handles: *mut *mut mlx_sys::mlx_array,
        v_scale_handles: *mut *mut mlx_sys::mlx_array,
        prefill_offset: i32,
    ) -> i32;

    fn mlx_gemma4_forward_paged(
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

    fn mlx_clear_weights();
    fn mlx_gemma4_reset();
}

/// Test 1 — pre-init guard: calling `_forward_paged` BEFORE
/// `_init_paged` must early-exit with `output_logits = nullptr` and
/// must not crash. Proves the new FFI symbol is linked and the
/// `g_gemma4_paged_inited` gate works.
#[test]
fn gemma4_forward_paged_before_init_returns_null_no_crash() {
    unsafe {
        mlx_gemma4_reset();
        mlx_clear_weights();
    }

    let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
    let mut offset_out: i32 = -1;

    unsafe {
        mlx_gemma4_forward_paged(
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
    assert_eq!(
        offset_out, -1,
        "offset_out must not be written when uninitialized"
    );
}

/// Test 2 — graph build smoke: register no weights, build all per-
/// layer pool / scale arrays, and call `_init_paged` then
/// `_forward_paged`. The forward will throw "Weight not found" inside
/// the compile graph; the C++ catch must turn that into a clean
/// `output_logits = nullptr` return, not a crash.
#[test]
fn gemma4_forward_paged_graph_builds_without_crash() {
    if !metal_available() {
        eprintln!(
            "skipping gemma4_forward_paged_graph_builds_without_crash: Metal unavailable on this host"
        );
        return;
    }
    unsafe {
        mlx_gemma4_reset();
        mlx_clear_weights();
    }

    let Some((mut k_vec, mut v_vec, mut ks_vec, mut vs_vec)) = build_layer_handles() else {
        eprintln!(
            "skipping gemma4_forward_paged_graph_builds_without_crash: array allocation failed (likely no Metal)"
        );
        return;
    };

    let init_status = unsafe {
        mlx_gemma4_init_paged(
            NUM_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            GLOBAL_NUM_KV_HEADS,
            GLOBAL_HEAD_DIM,
            ROPE_THETA,
            ROPE_LOCAL_BASE_FREQ,
            PARTIAL_ROTARY_FACTOR,
            RMS_NORM_EPS,
            SLIDING_WINDOW,
            TIE_EMBED,
            MAX_KV_LEN,
            BATCH_SIZE,
            NUM_EXPERTS,
            TOP_K_EXPERTS,
            MOE_INTERMEDIATE_SIZE,
            INTERMEDIATE_SIZE,
            FINAL_LOGIT_SOFTCAPPING,
            LAYER_TYPES.as_ptr(),
            LAYER_TYPES.len() as i32,
            k_vec.as_mut_ptr(),
            v_vec.as_mut_ptr(),
            ks_vec.as_mut_ptr(),
            vs_vec.as_mut_ptr(),
            0,
        )
    };
    assert_eq!(
        init_status, 0,
        "mlx_gemma4_init_paged must succeed with full per-layer handle bundle"
    );

    let offset_arr = i32_arr_from(&[0], &[1]);
    let block_table_data: Vec<i32> = vec![-1; MAX_BLOCKS_PER_SEQ as usize];
    let block_table = i32_arr_from(&block_table_data, &[1, MAX_BLOCKS_PER_SEQ]);
    let slot_mapping_data: Vec<i64> = vec![-1; CHUNK_SIZE_MAX as usize];
    let slot_mapping = i64_arr_from(&slot_mapping_data, &[CHUNK_SIZE_MAX]);
    let num_valid_tokens = i32_arr_from(&[1], &[1]);
    let num_valid_blocks = i32_arr_from(&[1], &[1]);
    let seq_lens = i32_arr_from(&[1], &[1]);
    let input_ids = i32_zeros(&[BATCH_SIZE as i64, 1]);
    let embedding_weight = bf16_zeros(&[1, HIDDEN_SIZE as i64]);

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
            "skipping gemma4_forward_paged_graph_builds_without_crash: input array allocation failed"
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
            release_layer_handles(&mut k_vec, &mut v_vec, &mut ks_vec, &mut vs_vec);
        }
        return;
    }

    let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
    let mut offset_out: i32 = -1;
    unsafe {
        mlx_gemma4_forward_paged(
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

    if !logits.is_null() {
        unsafe { delete(logits) };
    }

    unsafe {
        delete(input_ids);
        delete(embedding_weight);
        delete(offset_arr);
        delete(block_table);
        delete(slot_mapping);
        delete(num_valid_tokens);
        delete(num_valid_blocks);
        delete(seq_lens);
        release_layer_handles(&mut k_vec, &mut v_vec, &mut ks_vec, &mut vs_vec);
        mlx_gemma4_reset();
    }
}

/// Test 3 — `mlx_gemma4_reset` MUST clear the paged globals. Before
/// Phase 7 reset only cleared the legacy flat compile state. After
/// init flips `g_gemma4_paged_inited = true`, calling reset must
/// flip it back to false; subsequent `_forward_paged` calls then
/// hit the early-exit guard and return null without crashing.
#[test]
fn gemma4_forward_paged_after_reset_returns_null() {
    if !metal_available() {
        eprintln!("skipping gemma4_forward_paged_after_reset_returns_null: Metal unavailable");
        return;
    }
    unsafe {
        mlx_gemma4_reset();
        mlx_clear_weights();
    }

    let Some((mut k_vec, mut v_vec, mut ks_vec, mut vs_vec)) = build_layer_handles() else {
        eprintln!(
            "skipping gemma4_forward_paged_after_reset_returns_null: array allocation failed"
        );
        return;
    };

    unsafe {
        let init_status = mlx_gemma4_init_paged(
            NUM_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            GLOBAL_NUM_KV_HEADS,
            GLOBAL_HEAD_DIM,
            ROPE_THETA,
            ROPE_LOCAL_BASE_FREQ,
            PARTIAL_ROTARY_FACTOR,
            RMS_NORM_EPS,
            SLIDING_WINDOW,
            TIE_EMBED,
            MAX_KV_LEN,
            BATCH_SIZE,
            NUM_EXPERTS,
            TOP_K_EXPERTS,
            MOE_INTERMEDIATE_SIZE,
            INTERMEDIATE_SIZE,
            FINAL_LOGIT_SOFTCAPPING,
            LAYER_TYPES.as_ptr(),
            LAYER_TYPES.len() as i32,
            k_vec.as_mut_ptr(),
            v_vec.as_mut_ptr(),
            ks_vec.as_mut_ptr(),
            vs_vec.as_mut_ptr(),
            42, // arbitrary prefill_offset to make stale state visible
        );
        assert_eq!(init_status, 0, "init must succeed");

        // Reset must clear `g_gemma4_paged_inited`. After Phase 7 this
        // returns the paged path to "uninitialized" — same state as
        // before any init call.
        mlx_gemma4_reset();

        let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
        let mut offset_out: i32 = -1;
        mlx_gemma4_forward_paged(
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
            "forward_paged after reset must return null logits"
        );
        assert_eq!(offset_out, -1, "offset_out must not be written after reset");

        release_layer_handles(&mut k_vec, &mut v_vec, &mut ks_vec, &mut vs_vec);
    }
}

/// Test 4 — `mlx_gemma4_forward_paged` enforces the single-token
/// decode contract. Calling with `slot_mapping.shape == [2]` and a
/// 2-element `input_ids` must return null logits without crashing.
#[test]
fn gemma4_forward_paged_rejects_multi_token_contract_violation() {
    if !metal_available() {
        eprintln!(
            "skipping gemma4_forward_paged_rejects_multi_token_contract_violation: Metal unavailable"
        );
        return;
    }
    unsafe {
        mlx_gemma4_reset();
        mlx_clear_weights();
    }

    let Some((mut k_vec, mut v_vec, mut ks_vec, mut vs_vec)) = build_layer_handles() else {
        eprintln!(
            "skipping gemma4_forward_paged_rejects_multi_token_contract_violation: array allocation failed"
        );
        return;
    };

    let init_status = unsafe {
        mlx_gemma4_init_paged(
            NUM_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            GLOBAL_NUM_KV_HEADS,
            GLOBAL_HEAD_DIM,
            ROPE_THETA,
            ROPE_LOCAL_BASE_FREQ,
            PARTIAL_ROTARY_FACTOR,
            RMS_NORM_EPS,
            SLIDING_WINDOW,
            TIE_EMBED,
            MAX_KV_LEN,
            BATCH_SIZE,
            NUM_EXPERTS,
            TOP_K_EXPERTS,
            MOE_INTERMEDIATE_SIZE,
            INTERMEDIATE_SIZE,
            FINAL_LOGIT_SOFTCAPPING,
            LAYER_TYPES.as_ptr(),
            LAYER_TYPES.len() as i32,
            k_vec.as_mut_ptr(),
            v_vec.as_mut_ptr(),
            ks_vec.as_mut_ptr(),
            vs_vec.as_mut_ptr(),
            0,
        )
    };
    assert_eq!(init_status, 0, "init must succeed");

    let offset_arr = i32_arr_from(&[0], &[1]);
    let block_table_data: Vec<i32> = vec![-1; MAX_BLOCKS_PER_SEQ as usize];
    let block_table = i32_arr_from(&block_table_data, &[1, MAX_BLOCKS_PER_SEQ]);
    // 2 slots — violates the [1] contract.
    let bad_slot_mapping = i64_arr_from(&[-1, -1], &[2]);
    let num_valid_tokens = i32_arr_from(&[1], &[1]);
    let num_valid_blocks = i32_arr_from(&[1], &[1]);
    let seq_lens = i32_arr_from(&[1], &[1]);
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
            "skipping gemma4_forward_paged_rejects_multi_token_contract_violation: input array allocation failed"
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
            release_layer_handles(&mut k_vec, &mut v_vec, &mut ks_vec, &mut vs_vec);
        }
        return;
    }

    let mut logits: *mut mlx_sys::mlx_array = ptr::null_mut();
    let mut offset_out: i32 = -1;
    unsafe {
        mlx_gemma4_forward_paged(
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
        release_layer_handles(&mut k_vec, &mut v_vec, &mut ks_vec, &mut vs_vec);
        mlx_gemma4_reset();
    }
}
