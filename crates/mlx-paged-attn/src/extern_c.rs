//! `extern "C"` shim around the existing Rust Metal dispatch.
//!
//! These thin wrappers expose the `dispatch_reshape_and_cache_raw` and
//! `dispatch_paged_attention_auto` Rust dispatchers (in [`crate::metal`])
//! under a stable, name-mangle-free C ABI so the MLX `Custom` primitives
//! in `crates/mlx-sys/src/mlx_paged_ops.cpp` can call them from inside
//! `eval_gpu`.
//!
//! The wrappers accept raw `MTLBuffer*` pointers and scalar params, and
//! call straight into the existing Rust dispatch which uses
//! `mlx-paged-attn`'s separate `MetalState::command_queue` — i.e. we do
//! **not** share the MLX command queue. Callers MUST evaluate any prior
//! dependencies before invoking these wrappers (the C++ primitive's
//! `eval_gpu` does this via `mlx_metal_synchronize` /
//! `mlx::core::synchronize`).
//!
//! ## Calling contract
//!
//! All wrapper functions:
//! - Take raw `*mut c_void` Metal buffer pointers (extracted by the
//!   caller via `mlx_array_get_metal_buffer`).
//! - Take primitive scalar params by value.
//! - Block on `command_buffer.wait_until_completed()` inside the
//!   inner dispatcher (the existing dispatcher does this synchronously).
//! - Return `0` on success, `-1` on error. Errors are also written to
//!   `stderr` so callers can see them in test output.
//!
//! All pointer parameters MUST be valid `MTLBuffer*` pointers. The
//! caller is responsible for keeping the underlying MLX arrays alive
//! and evaluated until the wrapper returns.

#![cfg(target_os = "macos")]

use std::ffi::c_void;

use metal::Buffer;
use metal::foreign_types::ForeignType;

use crate::metal::{
    MetalDtype, PagedAttentionParams, PagedAttentionVarlenParams, RawBufferInfo,
    ReshapeAndCacheParams, dispatch_paged_attention_auto, dispatch_paged_attention_varlen_auto,
    dispatch_reshape_and_cache_raw,
};

/// `KvDtype` mirror for the C ABI. Must agree, value-by-value, with
/// the C++ `enum class KvDtype : uint8_t` declared in
/// `crates/mlx-sys/src/mlx_paged_ops.h`.
///
/// Keep these in lockstep — adding a variant on one side without the
/// other silently misroutes the kernel template selection.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtypeC {
    Fp16 = 0,
    Bf16 = 1,
    Fp8 = 2,
}

impl KvDtypeC {
    fn to_metal(self) -> MetalDtype {
        match self {
            KvDtypeC::Fp16 => MetalDtype::Float16,
            KvDtypeC::Bf16 => MetalDtype::BFloat16,
            KvDtypeC::Fp8 => MetalDtype::UChar,
        }
    }

    /// Whether this dtype represents an FP8 (1-byte) cache layout.
    fn is_fp8(self) -> bool {
        matches!(self, KvDtypeC::Fp8)
    }

    /// `x` factor for the vLLM K layout: 16 / sizeof(dtype).
    /// Used to validate the C++ side passed a consistent `x_pack`.
    fn expected_x(self) -> i32 {
        match self {
            KvDtypeC::Fp16 | KvDtypeC::Bf16 => 8,
            KvDtypeC::Fp8 => 16,
        }
    }
}

fn parse_kv_dtype(raw: u8) -> Option<KvDtypeC> {
    match raw {
        0 => Some(KvDtypeC::Fp16),
        1 => Some(KvDtypeC::Bf16),
        2 => Some(KvDtypeC::Fp8),
        _ => None,
    }
}

/// Borrow an `MTLBuffer*` (held by MLX) as an owned `Buffer` so the
/// existing dispatcher API can take it. The caller MUST `mem::forget`
/// the wrapper after the call so the underlying refcount isn't
/// decremented — MLX still owns the allocation.
///
/// # Safety
/// - `raw` must be a live `MTLBuffer*` (`id<MTLBuffer>` retained by
///   MLX) and remain valid until the caller forgets the returned
///   wrapper. Otherwise, the wrapper's `Drop` will decrement an
///   already-zero refcount and crash.
unsafe fn borrow_buffer(raw: *mut c_void) -> Buffer {
    // SAFETY: ForeignType::from_ptr is the documented round-trip from
    // a raw pointer. The wrapper takes ownership semantics; we
    // counterbalance with `mem::forget` at every call site.
    unsafe { Buffer::from_ptr(raw as *mut _) }
}

/// `extern "C"` wrapper around `dispatch_reshape_and_cache_raw`.
///
/// Writes a chunk of new K/V into the per-layer block-paged K/V pool.
/// The `key_pool` / `value_pool` buffers MUST already hold the layer's
/// pool storage (typically allocated by `LayerKVPool::new`).
///
/// # Buffer layout (matching the existing Rust dispatch)
/// - `key_pool`: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
/// - `value_pool`: `[num_blocks, num_kv_heads, head_size, block_size]`
/// - `new_keys`, `new_values`: `[num_tokens, num_kv_heads, head_size]`
/// - `slot_mapping`: `[num_tokens]` of `i64`
///
/// # Safety
/// - All pointer parameters MUST be valid `MTLBuffer*` pointers.
/// - The MLX arrays they point at MUST be evaluated (caller is
///   responsible — typically by `mlx_metal_synchronize` before the
///   call).
/// - Returns `0` on success, `-1` on any error (validation, dispatch,
///   or kernel pipeline build failure).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mlx_paged_attn_reshape_and_cache_dispatch(
    key_pool_buffer: *mut c_void,
    value_pool_buffer: *mut c_void,
    new_keys_buffer: *mut c_void,
    new_keys_offset: usize,
    new_values_buffer: *mut c_void,
    new_values_offset: usize,
    slot_mapping_buffer: *mut c_void,
    slot_mapping_offset: usize,
    num_tokens: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    x_pack: i32,
    kv_dtype_raw: u8,
    k_scale: f32,
    v_scale: f32,
) -> i32 {
    if key_pool_buffer.is_null()
        || value_pool_buffer.is_null()
        || new_keys_buffer.is_null()
        || new_values_buffer.is_null()
        || slot_mapping_buffer.is_null()
    {
        eprintln!("mlx_paged_attn_reshape_and_cache_dispatch: null buffer pointer");
        return -1;
    }
    let Some(kv_dtype) = parse_kv_dtype(kv_dtype_raw) else {
        eprintln!(
            "mlx_paged_attn_reshape_and_cache_dispatch: invalid kv_dtype_raw {}",
            kv_dtype_raw
        );
        return -1;
    };

    if x_pack != kv_dtype.expected_x() {
        eprintln!(
            "mlx_paged_attn_reshape_and_cache_dispatch: x_pack {} disagrees with \
             kv_dtype {:?} (expected {})",
            x_pack,
            kv_dtype,
            kv_dtype.expected_x()
        );
        return -1;
    }

    if num_tokens == 0 {
        // No-op write — let the caller's compile graph treat this as
        // a successful tail.
        return 0;
    }

    // io dtype = cache dtype for non-FP8, BF16 for FP8.
    let cache_dtype = kv_dtype.to_metal();
    let input_dtype = if kv_dtype.is_fp8() {
        MetalDtype::BFloat16
    } else {
        cache_dtype
    };

    let key_stride = (num_kv_heads * head_size) as i32;
    let value_stride = key_stride;

    let params = ReshapeAndCacheParams {
        num_tokens,
        num_heads: num_kv_heads,
        head_size,
        block_size,
        key_stride,
        value_stride,
        x: x_pack,
        k_scale,
        v_scale,
    };

    let new_keys = RawBufferInfo {
        ptr: new_keys_buffer,
        offset: new_keys_offset,
    };
    let new_values = RawBufferInfo {
        ptr: new_values_buffer,
        offset: new_values_offset,
    };
    let slot_mapping = RawBufferInfo {
        ptr: slot_mapping_buffer,
        offset: slot_mapping_offset,
    };

    // SAFETY: caller guarantees the pool buffers are live MTLBuffer*
    // pointers retained by MLX. We forget the owned wrappers after the
    // dispatch returns to avoid double-release.
    let key_pool = unsafe { borrow_buffer(key_pool_buffer) };
    let value_pool = unsafe { borrow_buffer(value_pool_buffer) };

    let result = unsafe {
        dispatch_reshape_and_cache_raw(
            &new_keys,
            &new_values,
            &key_pool,
            &value_pool,
            &slot_mapping,
            &params,
            input_dtype,
            cache_dtype,
        )
    };

    std::mem::forget(key_pool);
    std::mem::forget(value_pool);

    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("mlx_paged_attn_reshape_and_cache_dispatch: dispatch error: {e}");
            -1
        }
    }
}

/// `extern "C"` wrapper around `dispatch_paged_attention_auto`.
///
/// Computes attention with K/V gathered from block-paged storage via
/// `block_table` + `seq_lens`. Auto-picks V1/V2 based on
/// `max_context_len`.
///
/// # Buffer layout
/// - `queries`: `[num_seqs, num_q_heads, head_size]`
/// - `key_pool`: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
/// - `value_pool`: `[num_blocks, num_kv_heads, head_size, block_size]`
/// - `block_table`: `[num_seqs, max_blocks_per_seq]` `i32`
/// - `seq_lens`: `[num_seqs]` `i32`
/// - `output`: pre-allocated `[num_seqs, num_q_heads, head_size]` in io
///   dtype matching `kv_dtype` for non-FP8 / BF16 for FP8.
///
/// The output is allocated by the C++ side (via MLX `set_data`). This
/// shim blits the dispatcher's internal output buffer into the
/// caller-supplied output buffer.
///
/// # Safety
/// - All pointer parameters MUST be valid `MTLBuffer*` pointers.
/// - The MLX arrays they point at MUST be evaluated.
/// - `output_buffer` MUST be at least
///   `num_seqs * num_q_heads * head_size * sizeof(io_dtype)` bytes
///   starting at `output_offset`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mlx_paged_attn_paged_attention_dispatch(
    queries_buffer: *mut c_void,
    queries_offset: usize,
    key_pool_buffer: *mut c_void,
    value_pool_buffer: *mut c_void,
    block_table_buffer: *mut c_void,
    seq_lens_buffer: *mut c_void,
    output_buffer: *mut c_void,
    output_offset: usize,
    num_seqs: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    max_context_len: u32,
    max_blocks_per_seq: u32,
    scale: f32,
    softcap: f32,
    sliding_window: i32,
    kv_dtype_raw: u8,
    k_scale: f32,
    v_scale: f32,
) -> i32 {
    if queries_buffer.is_null()
        || key_pool_buffer.is_null()
        || value_pool_buffer.is_null()
        || block_table_buffer.is_null()
        || seq_lens_buffer.is_null()
        || output_buffer.is_null()
    {
        eprintln!("mlx_paged_attn_paged_attention_dispatch: null buffer pointer");
        return -1;
    }
    // Negative sliding_window is nonsensical (the only well-defined
    // sentinel for "no mask" is 0) and would otherwise reach the Metal
    // kernel as garbage, so reject it up front.
    if sliding_window < 0 {
        eprintln!(
            "mlx_paged_attn_paged_attention_dispatch: sliding_window={sliding_window} \
             must be >= 0 (use 0 to disable the sliding mask)"
        );
        return -1;
    }
    let Some(kv_dtype) = parse_kv_dtype(kv_dtype_raw) else {
        eprintln!(
            "mlx_paged_attn_paged_attention_dispatch: invalid kv_dtype_raw {}",
            kv_dtype_raw
        );
        return -1;
    };
    if num_seqs == 0 || num_q_heads == 0 || head_size == 0 {
        eprintln!("mlx_paged_attn_paged_attention_dispatch: zero-sized dispatch");
        return -1;
    }
    if max_context_len == 0 || max_blocks_per_seq == 0 {
        eprintln!("mlx_paged_attn_paged_attention_dispatch: zero context/blocks");
        return -1;
    }

    let cache_dtype = kv_dtype.to_metal();
    let io_dtype = if kv_dtype.is_fp8() {
        MetalDtype::BFloat16
    } else {
        cache_dtype
    };

    // softcap = 0.0 is the C++ caller's "disabled" sentinel; the kernel
    // expects 1.0 to mean disabled. Translate.
    let softcapping = if softcap == 0.0 { 1.0 } else { softcap };

    let q_stride = (num_q_heads * head_size) as i32;
    let kv_block_stride = (num_kv_heads * head_size * block_size) as i32;
    let kv_head_stride = (head_size * block_size) as i32;

    let params = PagedAttentionParams {
        num_seqs,
        num_heads: num_q_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_seq_len: max_context_len,
        max_num_blocks_per_seq: max_blocks_per_seq,
        scale,
        softcapping,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale,
        v_scale,
        sliding_window,
    };

    let queries_raw = RawBufferInfo {
        ptr: queries_buffer,
        offset: queries_offset,
    };

    // SAFETY: caller guarantees the pool/aux buffers are live MTLBuffer*
    // pointers retained by MLX.
    let key_pool = unsafe { borrow_buffer(key_pool_buffer) };
    let value_pool = unsafe { borrow_buffer(value_pool_buffer) };
    let block_table = unsafe { borrow_buffer(block_table_buffer) };
    let seq_lens = unsafe { borrow_buffer(seq_lens_buffer) };

    let dispatch_result = unsafe {
        dispatch_paged_attention_auto(
            &queries_raw,
            &key_pool,
            &value_pool,
            &block_table,
            &seq_lens,
            max_context_len,
            &params,
            io_dtype,
            cache_dtype,
        )
    };

    let blit_result = match dispatch_result {
        Ok(out) => {
            // SAFETY: caller guarantees output_buffer is live for the
            // call, and large enough to hold the output. We forget the
            // owned wrapper after the blit.
            let output_owned = unsafe { borrow_buffer(output_buffer) };
            let r = blit_attention_output(&out, &output_owned, output_offset, io_dtype);
            std::mem::forget(output_owned);
            r
        }
        Err(e) => Err(e),
    };

    std::mem::forget(key_pool);
    std::mem::forget(value_pool);
    std::mem::forget(block_table);
    std::mem::forget(seq_lens);

    match blit_result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("mlx_paged_attn_paged_attention_dispatch: error: {e}");
            -1
        }
    }
}

/// Blit a `PagedAttentionOutput` (allocated by the dispatcher) into the
/// caller-supplied output buffer at `output_offset` bytes.
fn blit_attention_output(
    out: &crate::metal::PagedAttentionOutput,
    output_buf: &Buffer,
    output_offset: usize,
    io_dtype: MetalDtype,
) -> Result<(), String> {
    let element_size = io_dtype.size();
    let total_bytes = out.num_elements() * element_size;

    let state = crate::metal::MetalState::get()?;
    let command_buffer = state.command_queue.new_command_buffer();
    let blit_encoder = command_buffer.new_blit_command_encoder();

    blit_encoder.copy_from_buffer(
        &out.buffer,
        0,
        output_buf,
        output_offset as u64,
        total_bytes as u64,
    );

    blit_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// `extern "C"` wrapper around `dispatch_paged_attention_varlen_auto`
/// (multi-row paged attention for MTP speculative decoding).
///
/// This is a SIBLING of `mlx_paged_attn_paged_attention_dispatch`, not a
/// replacement: the single-row entrypoint is the AR path; the MTP verify
/// path uses this varlen entrypoint.
///
/// # Buffer layout
/// - `queries`: `[total_queries, num_q_heads, head_size]` — ragged Q.
/// - `cu_seqlens_q`: `[num_seqs + 1]` int32 cumulative query counts.
///   `cu_seqlens_q[0]` must equal 0 and `cu_seqlens_q[num_seqs]` must
///   equal `total_queries`. Caller validates this.
/// - `key_pool`, `value_pool`, `block_table`, `seq_lens`, `output`: same
///   layouts as the single-row entrypoint.
///
/// # Safety
/// - All pointer parameters MUST be valid `MTLBuffer*` pointers.
/// - The MLX arrays they point at MUST be evaluated before the call.
/// - `output_buffer` MUST be at least
///   `total_queries * num_q_heads * head_size * sizeof(io_dtype)` bytes
///   starting at `output_offset`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mlx_paged_attn_paged_attention_varlen_dispatch(
    queries_buffer: *mut c_void,
    queries_offset: usize,
    key_pool_buffer: *mut c_void,
    value_pool_buffer: *mut c_void,
    block_table_buffer: *mut c_void,
    seq_lens_buffer: *mut c_void,
    cu_seqlens_q_buffer: *mut c_void,
    output_buffer: *mut c_void,
    output_offset: usize,
    num_seqs: u32,
    total_queries: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    max_context_len: u32,
    max_blocks_per_seq: u32,
    scale: f32,
    softcap: f32,
    sliding_window: i32,
    kv_dtype_raw: u8,
    k_scale: f32,
    v_scale: f32,
) -> i32 {
    if queries_buffer.is_null()
        || key_pool_buffer.is_null()
        || value_pool_buffer.is_null()
        || block_table_buffer.is_null()
        || seq_lens_buffer.is_null()
        || cu_seqlens_q_buffer.is_null()
        || output_buffer.is_null()
    {
        eprintln!("mlx_paged_attn_paged_attention_varlen_dispatch: null buffer pointer");
        return -1;
    }
    if sliding_window < 0 {
        eprintln!(
            "mlx_paged_attn_paged_attention_varlen_dispatch: sliding_window={sliding_window} \
             must be >= 0 (use 0 to disable the sliding mask)"
        );
        return -1;
    }
    let Some(kv_dtype) = parse_kv_dtype(kv_dtype_raw) else {
        eprintln!(
            "mlx_paged_attn_paged_attention_varlen_dispatch: invalid kv_dtype_raw {}",
            kv_dtype_raw
        );
        return -1;
    };
    if num_seqs == 0 || total_queries == 0 || num_q_heads == 0 || head_size == 0 {
        eprintln!("mlx_paged_attn_paged_attention_varlen_dispatch: zero-sized dispatch");
        return -1;
    }
    if max_context_len == 0 || max_blocks_per_seq == 0 {
        eprintln!("mlx_paged_attn_paged_attention_varlen_dispatch: zero context/blocks");
        return -1;
    }

    let cache_dtype = kv_dtype.to_metal();
    let io_dtype = if kv_dtype.is_fp8() {
        MetalDtype::BFloat16
    } else {
        cache_dtype
    };

    // softcap = 0.0 is the C++ "disabled" sentinel; kernel expects 1.0.
    let softcapping = if softcap == 0.0 { 1.0 } else { softcap };

    let q_stride = (num_q_heads * head_size) as i32;
    let kv_block_stride = (num_kv_heads * head_size * block_size) as i32;
    let kv_head_stride = (head_size * block_size) as i32;

    let params = PagedAttentionVarlenParams {
        num_seqs,
        total_queries,
        num_heads: num_q_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_seq_len: max_context_len,
        max_num_blocks_per_seq: max_blocks_per_seq,
        scale,
        softcapping,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale,
        v_scale,
        sliding_window,
    };

    let queries_raw = RawBufferInfo {
        ptr: queries_buffer,
        offset: queries_offset,
    };

    // SAFETY: caller guarantees the buffers are live MTLBuffer* pointers
    // retained by MLX. Forget the owned wrappers after the dispatch so
    // the underlying refcounts aren't double-decremented.
    let key_pool = unsafe { borrow_buffer(key_pool_buffer) };
    let value_pool = unsafe { borrow_buffer(value_pool_buffer) };
    let block_table = unsafe { borrow_buffer(block_table_buffer) };
    let seq_lens = unsafe { borrow_buffer(seq_lens_buffer) };
    let cu_seqlens_q = unsafe { borrow_buffer(cu_seqlens_q_buffer) };

    let dispatch_result = unsafe {
        dispatch_paged_attention_varlen_auto(
            &queries_raw,
            &key_pool,
            &value_pool,
            &block_table,
            &seq_lens,
            &cu_seqlens_q,
            max_context_len,
            &params,
            io_dtype,
            cache_dtype,
        )
    };

    let blit_result = match dispatch_result {
        Ok(out) => {
            // SAFETY: caller guarantees output_buffer is live and large
            // enough. Reuses the same blit helper as the single-row path.
            let output_owned = unsafe { borrow_buffer(output_buffer) };
            let r = blit_attention_output(&out, &output_owned, output_offset, io_dtype);
            std::mem::forget(output_owned);
            r
        }
        Err(e) => Err(e),
    };

    std::mem::forget(key_pool);
    std::mem::forget(value_pool);
    std::mem::forget(block_table);
    std::mem::forget(seq_lens);
    std::mem::forget(cu_seqlens_q);

    match blit_result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("mlx_paged_attn_paged_attention_varlen_dispatch: error: {e}");
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kv_dtype_roundtrip() {
        assert!(matches!(parse_kv_dtype(0), Some(KvDtypeC::Fp16)));
        assert!(matches!(parse_kv_dtype(1), Some(KvDtypeC::Bf16)));
        assert!(matches!(parse_kv_dtype(2), Some(KvDtypeC::Fp8)));
        assert!(parse_kv_dtype(3).is_none());
    }

    #[test]
    fn expected_x_for_each_dtype() {
        assert_eq!(KvDtypeC::Fp16.expected_x(), 8);
        assert_eq!(KvDtypeC::Bf16.expected_x(), 8);
        assert_eq!(KvDtypeC::Fp8.expected_x(), 16);
    }

    #[test]
    fn null_pointer_returns_error() {
        let rc = unsafe {
            mlx_paged_attn_reshape_and_cache_dispatch(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
                1,
                4,
                64,
                16,
                8,
                0,
                1.0,
                1.0,
            )
        };
        assert_eq!(rc, -1);
    }

    #[test]
    fn invalid_dtype_returns_error() {
        // Even with non-null pointers this should fail at the dtype
        // parse step. Use `1` as a placeholder pointer to force the
        // null check to pass.
        let dummy: *mut c_void = std::ptr::dangling_mut::<c_void>();
        let rc = unsafe {
            mlx_paged_attn_reshape_and_cache_dispatch(
                dummy, dummy, dummy, 0, dummy, 0, dummy, 0, 1, 4, 64, 16, 8, 99, 1.0, 1.0,
            )
        };
        assert_eq!(rc, -1);
    }

    #[test]
    fn x_pack_disagreement_returns_error() {
        let dummy: *mut c_void = std::ptr::dangling_mut::<c_void>();
        // Fp16 expects x_pack = 8; pass 16 to trigger the disagreement
        // check.
        let rc = unsafe {
            mlx_paged_attn_reshape_and_cache_dispatch(
                dummy, dummy, dummy, 0, dummy, 0, dummy, 0, 1, 4, 64, 16, 16, 0, 1.0, 1.0,
            )
        };
        assert_eq!(rc, -1);
    }
}
