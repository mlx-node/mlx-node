//! Phase 1 smoke tests for the new MLX paged-ops `Custom` primitives
//! (`PagedKVWrite`, `PagedAttention`) and the `extern "C"` shim that
//! bridges them to the existing `dispatch_*` Metal pipelines.
//!
//! ## Test list (matching the Phase-1 plan, + post-review additions)
//!
//! 1. **Round-trip K/V** — write 2 tokens of synthetic K/V into a
//!    miniature paged pool through the shim, read the pool back,
//!    and assert byte-equality at the right slot offsets.
//! 2. **Compile-trace cache key stability** — wraps the public
//!    `paged_kv_write` factory in `mlx::core::compile()` and verifies
//!    the inner trace function runs ONCE across two same-shape calls
//!    (i.e., the second call hits the compile cache). Implemented via
//!    a C++ FFI helper that exposes a static atomic trace counter.
//! 3. **FP8 scale plumbing** — runs a real FP8 kernel dispatch with
//!    non-default `k_scale`/`v_scale` and reads the resulting FP8
//!    bytes back to confirm scales propagated to the kernel (instead
//!    of being silently dropped). Also keeps the kernel-name
//!    selection sanity test as supplementary coverage.
//! 4. **`is_equivalent` correctness** — instantiate two
//!    `PagedKVWrite` primitives via the C++ FFI helper and assert
//!    `is_equivalent` correctly distinguishes equal vs. differing
//!    scalar state.
//! 5. **VJP throws** — instantiate a `PagedKVWrite` and a
//!    `PagedAttention` via the C++ FFI helper and assert each
//!    primitive's `vjp` raises `std::runtime_error`.
//! 6. **Output shape uses scalar state** — verify that
//!    `PagedAttention::output_shapes` reports
//!    `{q_num_tokens, num_q_heads, head_size}` from the primitive's
//!    scalar state (NOT echoing q's trailing dims).
//! 7. **Validation rejection paths** — the public `paged_attention`
//!    factory rejects (a) `sliding_window != 0`, (b) q whose trailing
//!    dims disagree with scalar state, (c) q rank != 3, (d)
//!    block_table batch / dtype mismatch, (e) seq_lens batch mismatch,
//!    (f) K/V pool inner-dim / x_pack / num_blocks mismatch. The
//!    public `paged_kv_write` factory rejects K/V pool shapes that
//!    disagree with scalar state, plus slot_mapping rank / dtype /
//!    length / out-of-range mismatches (Phase 1 eval-based bounds
//!    check). The Rust extern-C shim independently rejects nonzero
//!    `sliding_window` so a missing C++-side check can never tunnel
//!    through.
//!
//! All Metal-dependent tests gracefully skip on hosts where
//! `MetalState::get()` fails ("No Metal device found"). The non-Metal
//! tests run on every host that successfully linked the mlx-sys
//! library.

#![cfg(target_os = "macos")]

use std::ffi::c_void;

use metal::MTLResourceOptions;
use metal::foreign_types::ForeignType;

use mlx_paged_attn::metal::MetalState;
use mlx_paged_attn::mlx_paged_attn_reshape_and_cache_dispatch;

// =============================================================================
// Convenience: f32 → f16 / f32 → bf16 conversion (host-side, for test
// inputs).
// =============================================================================

fn f32_to_f16_bits(x: f32) -> u16 {
    // half crate isn't a workspace dep; do a simple manual cast that
    // covers normal positive values used in these tests.
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant_f32 = bits & 0x7fffff;
    if exp_f32 == 0xff {
        // Inf / NaN
        let mant = if mant_f32 != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1f << 10) | mant;
    }
    let exp_unbiased = exp_f32 - 127;
    if exp_unbiased < -14 {
        // Subnormal or zero (treat as zero for test convenience)
        return sign << 15;
    }
    if exp_unbiased > 15 {
        // Overflow → +/- inf
        return (sign << 15) | (0x1f << 10);
    }
    let exp_f16 = ((exp_unbiased + 15) as u16) & 0x1f;
    let mant_f16 = (mant_f32 >> 13) as u16;
    (sign << 15) | (exp_f16 << 10) | mant_f16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal — normalize.
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let f32_exp = ((127 - 15 + 1 + e) as u32) << 23;
        return f32::from_bits((sign << 31) | f32_exp | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | 0x7f80_0000 | (mant << 13));
    }
    let f32_exp = ((exp as i32 - 15 + 127) as u32) << 23;
    f32::from_bits((sign << 31) | f32_exp | (mant << 13))
}

// =============================================================================
// Test 1: round-trip K/V
//
// Build two miniature pools (key + value) sized for 4 blocks, 4 KV
// heads, head_size 64, block_size 16, FP16. Write 2 tokens of
// synthetic data through the shim, blit the pools back to host, and
// verify the values land at the right (block, head, position, slot)
// indices according to the kernel's K layout.
// =============================================================================

#[test]
fn round_trip_k_v_through_shim() {
    let state = match MetalState::get() {
        Ok(s) => s,
        Err(e) if e.contains("No Metal device found") => {
            eprintln!("skipping round_trip_k_v_through_shim: {e}");
            return;
        }
        Err(e) => panic!("unexpected MetalState::get failure: {e}"),
    };

    // Pool config: 4 blocks, 4 KV heads, head_size 64, block_size 16,
    // FP16 → x = 8.
    let num_blocks: u32 = 4;
    let num_kv_heads: u32 = 4;
    let head_size: u32 = 64;
    let block_size: u32 = 16;
    let x: u32 = 8;
    let element_size: u64 = 2;

    // Allocate K/V cache buffers in shared storage so we can read the
    // pool back without an extra blit.
    let key_cache_size = (num_blocks as u64)
        * (num_kv_heads as u64)
        * (head_size as u64 / x as u64)
        * (block_size as u64)
        * (x as u64)
        * element_size;
    let value_cache_size = (num_blocks as u64)
        * (num_kv_heads as u64)
        * (head_size as u64)
        * (block_size as u64)
        * element_size;

    let key_pool = state
        .device
        .new_buffer(key_cache_size, MTLResourceOptions::StorageModeShared);
    let value_pool = state
        .device
        .new_buffer(value_cache_size, MTLResourceOptions::StorageModeShared);

    // Zero-initialize pools so we can detect what got written.
    unsafe {
        std::ptr::write_bytes(key_pool.contents() as *mut u8, 0, key_cache_size as usize);
        std::ptr::write_bytes(
            value_pool.contents() as *mut u8,
            0,
            value_cache_size as usize,
        );
    }

    // 2 tokens of synthetic data.
    let num_tokens: u32 = 2;
    let tokens_size_elements =
        (num_tokens as usize) * (num_kv_heads as usize) * (head_size as usize);
    let mut new_k_host: Vec<u16> = Vec::with_capacity(tokens_size_elements);
    let mut new_v_host: Vec<u16> = Vec::with_capacity(tokens_size_elements);
    for t in 0..num_tokens {
        for h in 0..num_kv_heads {
            for j in 0..head_size {
                let k_val = (t as f32) * 1000.0 + (h as f32) * 100.0 + (j as f32);
                let v_val = -k_val;
                new_k_host.push(f32_to_f16_bits(k_val));
                new_v_host.push(f32_to_f16_bits(v_val));
            }
        }
    }
    let new_k = state.device.new_buffer_with_data(
        new_k_host.as_ptr() as *const _,
        (tokens_size_elements * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let new_v = state.device.new_buffer_with_data(
        new_v_host.as_ptr() as *const _,
        (tokens_size_elements * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Slot mapping: token 0 → slot 5 (block 0, position 5),
    //               token 1 → slot 21 (block 1, position 5).
    let slot_mapping_host: Vec<i64> = vec![5, 21];
    let slot_mapping = state.device.new_buffer_with_data(
        slot_mapping_host.as_ptr() as *const _,
        (slot_mapping_host.len() * std::mem::size_of::<i64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Dispatch through the shim.
    let rc = unsafe {
        mlx_paged_attn_reshape_and_cache_dispatch(
            key_pool.as_ptr() as *mut c_void,
            value_pool.as_ptr() as *mut c_void,
            new_k.as_ptr() as *mut c_void,
            0,
            new_v.as_ptr() as *mut c_void,
            0,
            slot_mapping.as_ptr() as *mut c_void,
            0,
            num_tokens,
            num_kv_heads,
            head_size,
            block_size,
            x as i32,
            0, // KvDtypeC::Fp16
            1.0,
            1.0,
        )
    };
    assert_eq!(rc, 0, "shim dispatch must succeed (rc=0)");

    // Read the pool buffers back as f16 bits.
    let key_bits: &[u16] = unsafe {
        std::slice::from_raw_parts(
            key_pool.contents() as *const u16,
            (key_cache_size / 2) as usize,
        )
    };
    let value_bits: &[u16] = unsafe {
        std::slice::from_raw_parts(
            value_pool.contents() as *const u16,
            (value_cache_size / 2) as usize,
        )
    };

    // Verify each (token, head, j) lands in the right slot.
    // K layout: key_cache[block_idx, head_idx, j/x, block_offset, j%x]
    // V layout: value_cache[block_idx, head_idx, j, block_offset]
    // strides (in elements):
    let head_per_block_k = (head_size / x) * block_size * x;
    let stride_block_k = num_kv_heads * head_per_block_k;
    let stride_head_k = head_per_block_k;
    let stride_xidx_k = block_size * x;
    let stride_blockoff_k = x;

    let head_per_block_v = head_size * block_size;
    let stride_block_v = num_kv_heads * head_per_block_v;
    let stride_head_v = head_per_block_v;
    let stride_j_v = block_size;

    for t in 0..num_tokens {
        let slot_idx = slot_mapping_host[t as usize];
        let block_idx = (slot_idx / block_size as i64) as u32;
        let block_offset = (slot_idx % block_size as i64) as u32;
        for h in 0..num_kv_heads {
            for j in 0..head_size {
                let x_idx = j / x;
                let x_offset = j % x;
                let k_target_idx = (block_idx * stride_block_k
                    + h * stride_head_k
                    + x_idx * stride_xidx_k
                    + block_offset * stride_blockoff_k
                    + x_offset) as usize;
                let v_target_idx = (block_idx * stride_block_v
                    + h * stride_head_v
                    + j * stride_j_v
                    + block_offset) as usize;

                let expected_k = (t as f32) * 1000.0 + (h as f32) * 100.0 + (j as f32);
                let expected_v = -expected_k;

                let actual_k = f16_bits_to_f32(key_bits[k_target_idx]);
                let actual_v = f16_bits_to_f32(value_bits[v_target_idx]);

                let kdiff = (actual_k - expected_k).abs();
                let vdiff = (actual_v - expected_v).abs();

                // F16 tolerance: ~1 ULP at small magnitudes; the
                // synthetic values can hit ~1300 which is fine in
                // F16 precision.
                let tol = expected_k.abs().max(1.0) * 1e-3;
                assert!(
                    kdiff <= tol,
                    "K mismatch at token={t} head={h} j={j}: \
                     expected {expected_k}, got {actual_k} (diff {kdiff})"
                );
                assert!(
                    vdiff <= tol,
                    "V mismatch at token={t} head={h} j={j}: \
                     expected {expected_v}, got {actual_v} (diff {vdiff})"
                );
            }
        }
    }
}

// =============================================================================
// Test 3: FP8 scale plumbing
//
// We can't easily round-trip FP8 (E4M3 dequant requires the kernel
// scale machinery and reading host-side FP8 bytes is value-dependent).
// Instead, this test asserts:
//   a) The shim REJECTS an x_pack disagreement for FP8 (`x_pack=8`
//      with `KvDtypeC::Fp8` should error since Fp8 expects x=16).
//   b) The shim ACCEPTS the correct (FP8, x_pack=16) combo and would
//      route to the FP8 kernel name. We don't actually fire the
//      kernel because that requires real quantization-aware K/V; the
//      kernel-name routing is verified in `state.rs`'s own tests
//      (`reshape_and_cache_kernel_name` returns the `_fp8` variant
//      for `(*, UChar)`).
//
// This is a "plumbing" test, not an end-to-end FP8 dispatch — Phase 1
// only needs the param to flow through correctly.
// =============================================================================

#[test]
fn fp8_scale_plumbing_rejects_wrong_x_pack() {
    // Use dummy non-null pointers; the shim's validation should
    // reject before it dereferences them.
    let dummy: *mut c_void = std::ptr::dangling_mut::<c_void>();

    // FP8 with x_pack = 8 is a contradiction (FP8 requires x=16).
    let rc = unsafe {
        mlx_paged_attn_reshape_and_cache_dispatch(
            dummy, dummy, dummy, 0, dummy, 0, dummy, 0, 1, 4, 64, 16, /*x_pack=*/ 8,
            /*kv_dtype=Fp8*/ 2, 0.5, 0.25,
        )
    };
    assert_eq!(rc, -1, "FP8 with x_pack=8 must be rejected");

    // Bf16 with x_pack = 16 is the inverse contradiction.
    let rc2 = unsafe {
        mlx_paged_attn_reshape_and_cache_dispatch(
            dummy, dummy, dummy, 0, dummy, 0, dummy, 0, 1, 4, 64, 16, /*x_pack=*/ 16,
            /*kv_dtype=Bf16*/ 1, 1.0, 1.0,
        )
    };
    assert_eq!(rc2, -1, "Bf16 with x_pack=16 must be rejected");
}

/// Verifies the FP8 kernel-name selection logic that `MetalState`
/// drives — combined with the shim's strict `(kv_dtype, x_pack)`
/// pairing check, this proves the FP8 dispatch path is wired through
/// to the correct Metal kernel instantiation.
///
/// Supplementary to `fp8_dispatch_uses_runtime_scales` below — this
/// kernel-name test catches a regression where the kernel suffix is
/// wired wrong, while the dispatch test catches a regression where
/// the suffix is right but the scales drop on the floor.
#[test]
fn fp8_kernel_name_selected_for_fp8_dtype() {
    use mlx_paged_attn::metal::MetalDtype;

    // Non-FP8 → name has no `_fp8` suffix.
    let bf16 = MetalState::reshape_and_cache_kernel_name(
        MetalDtype::BFloat16,
        MetalDtype::BFloat16,
        false,
    );
    assert!(!bf16.contains("_fp8"));
    assert!(bf16.contains("bfloat16_t"));

    // FP8 cache → name has `_fp8` suffix.
    let fp8 =
        MetalState::reshape_and_cache_kernel_name(MetalDtype::BFloat16, MetalDtype::UChar, true);
    assert!(fp8.ends_with("_fp8"));
    assert!(fp8.contains("uchar"));
}

/// Convert a positive single-precision float to its FP8 E4M3 byte
/// representation. Mirrors `metal/float8.metal::float_to_fp8_e4m3`
/// closely enough for the values used in this test (round to nearest
/// even, no NaN/Inf paths needed).
fn f32_to_fp8_e4m3_byte(f: f32) -> u8 {
    if f == 0.0 {
        return 0;
    }
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let abs = bits & 0x7fff_ffff;
    if abs >= 0x7f80_0000 {
        // Inf / NaN — mirror metal saturate behavior.
        let mant = if abs != 0x7f80_0000 { 1 } else { 0 };
        return ((sign << 7) | (0xf << 3) | mant) as u8;
    }
    let e = (((abs >> 23) & 0xff) as i32) - 127;
    let m = abs & 0x7f_ffff;
    const EXP_BITS: i32 = 4;
    const MAN_BITS: i32 = 3;
    const BIAS: i32 = 7;
    const EXP_MAX: i32 = (1 << EXP_BITS) - 2; // 14
    let e_fp8 = e + BIAS;
    if (1..=EXP_MAX).contains(&e_fp8) {
        let shift = 23 - MAN_BITS; // 20
        let mut mant = m >> shift;
        let lsb = mant & 1;
        let round = (m >> (shift - 1)) & 1;
        let sticky = u32::from((m & ((1u32 << (shift - 1)) - 1)) != 0);
        mant += round & (sticky | lsb);
        if mant >> MAN_BITS != 0 {
            // mantissa overflow
            let e2 = e_fp8 + 1;
            if e2 > EXP_MAX {
                return ((sign << 7) | (((1u32 << EXP_BITS) - 1) << MAN_BITS)) as u8;
            }
            return ((sign << 7) | ((e2 as u32) << MAN_BITS)) as u8;
        }
        return ((sign << 7) | ((e_fp8 as u32) << MAN_BITS) | (mant & ((1u32 << MAN_BITS) - 1)))
            as u8;
    }
    if e_fp8 < 1 - MAN_BITS {
        return (sign << 7) as u8;
    }
    // sub-normal
    let rshift = (1 - e_fp8) + (23 - MAN_BITS);
    let mant_full = 0x80_0000 | m;
    let rounded = (mant_full + (1 << (rshift - 1))) >> rshift;
    if rounded == 0 {
        return (sign << 7) as u8;
    }
    ((sign << 7) | (rounded & ((1u32 << MAN_BITS) - 1))) as u8
}

/// Convert f32 to bf16 (truncation — drops 16 LSBs).
fn f32_to_bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Real FP8 kernel dispatch with non-default scales.
///
/// Writes K=0.5 BF16, V=0.25 BF16 with `k_scale=0.5`, `v_scale=0.25`.
/// The kernel computes `to_fp8(K / k_scale) = to_fp8(1.0) = 0x38`
/// (E4M3, exp=7, mant=0). If the shim silently dropped the scales
/// (i.e., used 1.0), the cache would hold `to_fp8(0.5)` (= 0x30) for
/// K and `to_fp8(0.25)` (= 0x28) for V — distinct bytes. Reading the
/// cache bytes back distinguishes the two cases.
///
/// This is the canonical "did the scale plumbing actually work?"
/// observability test that the previous Phase-1 review found missing.
#[test]
fn fp8_dispatch_uses_runtime_scales() {
    let state = match MetalState::get() {
        Ok(s) => s,
        Err(e) if e.contains("No Metal device found") => {
            eprintln!("skipping fp8_dispatch_uses_runtime_scales: {e}");
            return;
        }
        Err(e) => panic!("unexpected MetalState::get failure: {e}"),
    };

    // Pool config: 4 blocks, 4 KV heads, head_size 64, block_size 16,
    // FP8 → x = 16. element_size = 1 byte (FP8 is `uchar` on cache).
    let num_blocks: u32 = 4;
    let num_kv_heads: u32 = 4;
    let head_size: u32 = 64;
    let block_size: u32 = 16;
    let x: u32 = 16;
    let cache_element_size: u64 = 1; // FP8 is 1 byte
    let input_element_size: u64 = 2; // BF16 inputs

    let key_cache_size = (num_blocks as u64)
        * (num_kv_heads as u64)
        * (head_size as u64 / x as u64)
        * (block_size as u64)
        * (x as u64)
        * cache_element_size;
    let value_cache_size = (num_blocks as u64)
        * (num_kv_heads as u64)
        * (head_size as u64)
        * (block_size as u64)
        * cache_element_size;

    let key_pool = state
        .device
        .new_buffer(key_cache_size, MTLResourceOptions::StorageModeShared);
    let value_pool = state
        .device
        .new_buffer(value_cache_size, MTLResourceOptions::StorageModeShared);

    // Mark every cache byte with a sentinel (0xff) so we can tell
    // which slots actually got written.
    unsafe {
        std::ptr::write_bytes(
            key_pool.contents() as *mut u8,
            0xff,
            key_cache_size as usize,
        );
        std::ptr::write_bytes(
            value_pool.contents() as *mut u8,
            0xff,
            value_cache_size as usize,
        );
    }

    // 2 tokens of synthetic data — uniform K=0.5, V=0.25.
    let num_tokens: u32 = 2;
    let tokens_size_elements =
        (num_tokens as usize) * (num_kv_heads as usize) * (head_size as usize);
    let k_value_f32: f32 = 0.5;
    let v_value_f32: f32 = 0.25;
    let k_bf16 = f32_to_bf16_bits(k_value_f32);
    let v_bf16 = f32_to_bf16_bits(v_value_f32);
    let new_k_host: Vec<u16> = vec![k_bf16; tokens_size_elements];
    let new_v_host: Vec<u16> = vec![v_bf16; tokens_size_elements];

    let new_k = state.device.new_buffer_with_data(
        new_k_host.as_ptr() as *const _,
        (tokens_size_elements as u64) * input_element_size,
        MTLResourceOptions::StorageModeShared,
    );
    let new_v = state.device.new_buffer_with_data(
        new_v_host.as_ptr() as *const _,
        (tokens_size_elements as u64) * input_element_size,
        MTLResourceOptions::StorageModeShared,
    );

    // Slot mapping: token 0 → slot 0, token 1 → slot 1 (block 0,
    // positions 0/1).
    let slot_mapping_host: Vec<i64> = vec![0, 1];
    let slot_mapping = state.device.new_buffer_with_data(
        slot_mapping_host.as_ptr() as *const _,
        (slot_mapping_host.len() * std::mem::size_of::<i64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Non-default scales chosen so K/k_scale = V/v_scale = 1.0.
    let k_scale: f32 = 0.5;
    let v_scale: f32 = 0.25;

    // Dispatch with FP8 + x=16 + non-default scales.
    let rc = unsafe {
        mlx_paged_attn_reshape_and_cache_dispatch(
            key_pool.as_ptr() as *mut c_void,
            value_pool.as_ptr() as *mut c_void,
            new_k.as_ptr() as *mut c_void,
            0,
            new_v.as_ptr() as *mut c_void,
            0,
            slot_mapping.as_ptr() as *mut c_void,
            0,
            num_tokens,
            num_kv_heads,
            head_size,
            block_size,
            x as i32,
            2, // KvDtypeC::Fp8
            k_scale,
            v_scale,
        )
    };
    assert_eq!(rc, 0, "FP8 dispatch must succeed (rc={rc})");

    // Read the FP8 bytes back. Expected:
    //   to_fp8_e4m3(K / k_scale) = to_fp8_e4m3(0.5/0.5) = to_fp8_e4m3(1.0)
    //   to_fp8_e4m3(V / v_scale) = to_fp8_e4m3(0.25/0.25) = to_fp8_e4m3(1.0)
    // If the scales were dropped (treated as 1.0):
    //   K_bytes would equal to_fp8_e4m3(0.5) = 0x30 (NOT 0x38).
    //   V_bytes would equal to_fp8_e4m3(0.25) = 0x28 (NOT 0x38).
    let expected_k_byte = f32_to_fp8_e4m3_byte(1.0);
    let expected_v_byte = f32_to_fp8_e4m3_byte(1.0);
    let bad_if_scale_dropped_k = f32_to_fp8_e4m3_byte(0.5);
    let bad_if_scale_dropped_v = f32_to_fp8_e4m3_byte(0.25);
    // Sanity: the two must differ — otherwise the test wouldn't
    // distinguish the two cases.
    assert_ne!(
        expected_k_byte, bad_if_scale_dropped_k,
        "FP8 bytes for 1.0 vs 0.5 must differ to distinguish scale-applied vs scale-dropped"
    );
    assert_ne!(expected_v_byte, bad_if_scale_dropped_v);

    let key_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(key_pool.contents() as *const u8, key_cache_size as usize)
    };
    let value_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            value_pool.contents() as *const u8,
            value_cache_size as usize,
        )
    };

    // K layout for FP8 (still vLLM):
    // key_cache[block_idx, head_idx, j/x, block_offset, j%x]
    let head_per_block_k = (head_size / x) * block_size * x;
    let stride_block_k = num_kv_heads * head_per_block_k;
    let stride_head_k = head_per_block_k;
    let stride_xidx_k = block_size * x;
    let stride_blockoff_k = x;

    let head_per_block_v = head_size * block_size;
    let stride_block_v = num_kv_heads * head_per_block_v;
    let stride_head_v = head_per_block_v;
    let stride_j_v = block_size;

    // Spot-check the (token=0, head=0, j=0) slot for both K and V
    // — and a deeper position to confirm the kernel wrote across
    // all heads/positions, not just head 0.
    for t in 0..num_tokens {
        let slot_idx = slot_mapping_host[t as usize];
        let block_idx = (slot_idx / block_size as i64) as u32;
        let block_offset = (slot_idx % block_size as i64) as u32;
        for h in [0u32, 3u32] {
            for j in [0u32, 32u32, 63u32] {
                let x_idx = j / x;
                let x_offset = j % x;
                let k_idx = (block_idx * stride_block_k
                    + h * stride_head_k
                    + x_idx * stride_xidx_k
                    + block_offset * stride_blockoff_k
                    + x_offset) as usize;
                let v_idx = (block_idx * stride_block_v
                    + h * stride_head_v
                    + j * stride_j_v
                    + block_offset) as usize;
                let actual_k = key_bytes[k_idx];
                let actual_v = value_bytes[v_idx];
                assert_eq!(
                    actual_k, expected_k_byte,
                    "K FP8 byte at token={t} head={h} j={j}: \
                     expected 0x{expected_k_byte:02x} (= to_fp8(1.0), proves scale was applied) \
                     but got 0x{actual_k:02x}. \
                     If got 0x{bad_if_scale_dropped_k:02x} the kernel ignored k_scale."
                );
                assert_eq!(
                    actual_v, expected_v_byte,
                    "V FP8 byte at token={t} head={h} j={j}: \
                     expected 0x{expected_v_byte:02x} (= to_fp8(1.0), proves scale was applied) \
                     but got 0x{actual_v:02x}. \
                     If got 0x{bad_if_scale_dropped_v:02x} the kernel ignored v_scale."
                );
            }
        }
    }
}

/// Negative test — the C++ layer's `paged_attention` shim must reject
/// zero context length so we don't tunnel that into the kernel
/// dispatcher. Provides coverage for the wire that connects the C++
/// primitive's `eval_gpu` to the Rust shim's parameter validation.
#[test]
fn paged_attention_shim_rejects_zero_context() {
    let dummy: *mut c_void = std::ptr::dangling_mut::<c_void>();
    let rc = unsafe {
        mlx_paged_attn::mlx_paged_attn_paged_attention_dispatch(
            dummy, 0, dummy, dummy, dummy, dummy, dummy, 0, /*num_seqs=*/ 1,
            /*num_q_heads=*/ 8, /*num_kv_heads=*/ 4, /*head_size=*/ 64,
            /*block_size=*/ 16, /*max_context_len=*/ 0, /*max_blocks_per_seq=*/ 4,
            /*scale=*/ 0.125, /*softcap=*/ 0.0, /*sliding_window=*/ 0,
            /*kv_dtype=Bf16*/ 1, /*k_scale=*/ 1.0, /*v_scale=*/ 1.0,
        )
    };
    assert_eq!(rc, -1, "max_context_len=0 must be rejected");
}

// =============================================================================
// Test 4: `is_equivalent` correctness
//
// Calls into the C++ FFI helper `mlx_paged_kv_write_is_equivalent`
// (in `mlx_paged_ops.cpp`) which constructs two `PagedKVWrite`
// primitives and reports the result of `lhs.is_equivalent(rhs)`.
// =============================================================================

#[test]
fn paged_kv_write_is_equivalent_same_state() {
    let same = unsafe {
        mlx_sys::mlx_paged_kv_write_is_equivalent(
            16, 4, 64, 8, 1, // KvDtype::Bf16
            16, 4, 64, 8, 1, // KvDtype::Bf16
        )
    };
    assert!(
        same,
        "primitives with identical scalar state must be equivalent"
    );
}

#[test]
fn paged_kv_write_is_equivalent_differing_block_size() {
    let diff_block_size = unsafe {
        mlx_sys::mlx_paged_kv_write_is_equivalent(
            16, 4, 64, 8, 1, // block_size=16
            32, 4, 64, 8, 1, // block_size=32
        )
    };
    assert!(
        !diff_block_size,
        "primitives differing in block_size must NOT be equivalent"
    );
}

#[test]
fn paged_kv_write_is_equivalent_differing_kv_dtype() {
    let diff_kv_dtype = unsafe {
        mlx_sys::mlx_paged_kv_write_is_equivalent(
            16, 4, 64, 8, 0, // KvDtype::Fp16
            16, 4, 64, 8, 1, // KvDtype::Bf16
        )
    };
    assert!(
        !diff_kv_dtype,
        "primitives differing in kv_dtype must NOT be equivalent"
    );
}

#[test]
fn paged_kv_write_is_equivalent_differing_num_kv_heads() {
    let diff = unsafe {
        mlx_sys::mlx_paged_kv_write_is_equivalent(
            16, 4, 64, 8, 1, // num_kv_heads=4
            16, 8, 64, 8, 1, // num_kv_heads=8
        )
    };
    assert!(!diff);
}

#[test]
fn paged_kv_write_is_equivalent_differing_head_size() {
    let diff = unsafe {
        mlx_sys::mlx_paged_kv_write_is_equivalent(
            16, 4, 64, 8, 1, // head_size=64
            16, 4, 128, 8, 1, // head_size=128
        )
    };
    assert!(!diff);
}

#[test]
fn paged_kv_write_is_equivalent_differing_x_pack() {
    let diff = unsafe {
        mlx_sys::mlx_paged_kv_write_is_equivalent(
            16, 4, 64, 8, 1, // x_pack=8
            16, 4, 64, 4, 1, // x_pack=4
        )
    };
    assert!(!diff);
}

#[test]
fn paged_attention_is_equivalent_same_state() {
    let same = unsafe {
        mlx_sys::mlx_paged_attention_is_equivalent(
            0.125, 0.0, 16, 8, 4, 64, 0, 1, // KvDtype::Bf16
            0.125, 0.0, 16, 8, 4, 64, 0, 1,
        )
    };
    assert!(same);
}

#[test]
fn paged_attention_is_equivalent_differing_scale() {
    let diff = unsafe {
        mlx_sys::mlx_paged_attention_is_equivalent(
            0.125, 0.0, 16, 8, 4, 64, 0, 1, // scale=0.125
            0.0625, 0.0, 16, 8, 4, 64, 0, 1, // scale=0.0625
        )
    };
    assert!(!diff, "differing scale must NOT be equivalent");
}

#[test]
fn paged_attention_is_equivalent_differing_sliding_window() {
    let diff = unsafe {
        mlx_sys::mlx_paged_attention_is_equivalent(
            0.125, 0.0, 16, 8, 4, 64, 0, 1, // sliding=0
            0.125, 0.0, 16, 8, 4, 64, 4096, 1, // sliding=4096
        )
    };
    assert!(!diff);
}

// =============================================================================
// Test 5: VJP throws
//
// Calls into the C++ FFI helpers that invoke `vjp` on a
// `PagedKVWrite` / `PagedAttention` primitive and report whether a
// `std::runtime_error` was thrown. The shim returns 1 on throw, 0
// otherwise.
// =============================================================================

#[test]
fn paged_kv_write_vjp_throws() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_vjp_throws() };
    assert_eq!(
        threw, 1,
        "PagedKVWrite::vjp must throw std::runtime_error (got {threw})"
    );
}

#[test]
fn paged_attention_vjp_throws() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_vjp_throws() };
    assert_eq!(
        threw, 1,
        "PagedAttention::vjp must throw std::runtime_error (got {threw})"
    );
}

// =============================================================================
// Test 6: `PagedAttention::output_shapes` reports scalar-state shape.
//
// MLX uses output_shapes to size the output buffer during compile
// replay. If `output_shapes` echoes q.shape() (which can disagree
// with the primitive's scalar state on the trailing dims) instead of
// `{q_num_tokens, num_q_heads, head_size}` from state, MLX would
// allocate a buffer of the wrong size — the kernel would then
// under- or over-write. This test verifies the spec.
// =============================================================================

#[test]
fn paged_attention_output_shapes_uses_scalar_state() {
    // Pass a q with deliberately-mismatched trailing dims (16 / 16
    // instead of num_q_heads=8 / head_size=64) and verify
    // `output_shapes` returns the SCALAR-state shape, not q's.
    let mut out_shape: [i32; 3] = [-1, -1, -1];
    let ndim = unsafe {
        mlx_sys::mlx_paged_attention_test_output_shapes(
            /*q_num_tokens=*/ 5,
            /*q_dim1_actual=*/ 16, // disagrees with num_q_heads=8 below
            /*q_dim2_actual=*/ 16, // disagrees with head_size=64 below
            /*scale=*/ 0.125,
            /*softcap=*/ 0.0,
            /*block_size=*/ 16,
            /*num_q_heads=*/ 8,
            /*num_kv_heads=*/ 4,
            /*head_size=*/ 64,
            /*sliding_window=*/ 0,
            /*kv_dtype_raw=*/ 1, // Bf16
            out_shape.as_mut_ptr(),
        )
    };
    assert_eq!(ndim, 3, "output shape must be rank 3");
    assert_eq!(
        out_shape,
        [5, 8, 64],
        "output_shapes must report {{q_num_tokens, num_q_heads, head_size}} \
         from scalar state — got {:?} which means q's trailing dims leaked through",
        out_shape
    );
}

// =============================================================================
// Test 7: validation rejection paths.
//
// The public `paged_attention` and `paged_kv_write` factories must
// reject inputs that would silently corrupt the kernel dispatch:
//   - sliding_window < 0 (Phase 7 lifts the Phase 1 sliding_window=0
//     restriction; only negative values remain illegal)
//   - q whose trailing dims disagree with primitive scalar state
//   - K/V pool whose interior dims disagree with primitive state
// The Rust extern-C shim ALSO rejects negative sliding_window so a
// missing C++-side check can't tunnel through.
// =============================================================================

#[test]
fn paged_attention_factory_rejects_sliding_window() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_sliding_window() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must throw std::invalid_argument when \
         sliding_window < 0 (got {threw}); Phase 7 lifts the Phase 1 \
         sliding_window=0 restriction but negative values remain illegal"
    );
}

#[test]
fn paged_attention_factory_rejects_q_shape_mismatch() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_q_shape_mismatch() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must throw when q's trailing dims disagree with \
         primitive scalar state (got {threw})"
    );
}

#[test]
fn paged_kv_write_factory_rejects_pool_shape_mismatch() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_pool_shape_mismatch() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must throw when K-pool interior dims disagree with \
         primitive scalar state (got {threw})"
    );
}

#[test]
fn paged_attention_shim_rejects_sliding_window() {
    // Independent extern-C-side guard: even if some caller bypasses
    // the C++ factory's rejection (e.g. constructs a primitive
    // directly) the shim will refuse a NEGATIVE sliding_window.
    // Phase 7 lifts the Phase 1 sliding_window=0 restriction; only
    // negative values remain illegal (the only "no mask" sentinel is
    // 0).
    let dummy: *mut c_void = std::ptr::dangling_mut::<c_void>();
    let rc = unsafe {
        mlx_paged_attn::mlx_paged_attn_paged_attention_dispatch(
            dummy, 0, dummy, dummy, dummy, dummy, dummy, 0, /*num_seqs=*/ 1,
            /*num_q_heads=*/ 8, /*num_kv_heads=*/ 4, /*head_size=*/ 64,
            /*block_size=*/ 16, /*max_context_len=*/ 32, /*max_blocks_per_seq=*/ 4,
            /*scale=*/ 0.125, /*softcap=*/ 0.0, /*sliding_window=*/ -1,
            /*kv_dtype=Bf16*/ 1, /*k_scale=*/ 1.0, /*v_scale=*/ 1.0,
        )
    };
    assert_eq!(rc, -1, "shim must reject sliding_window < 0");
}

// =============================================================================
// Test 2: compile-trace cache key stability.
//
// Wraps the public `paged_kv_write` factory in `mlx::core::compile()`
// (via the `mlx_paged_kv_write_compile_trace_smoke` C++ helper) and
// asserts that calling the compiled function twice with the same
// input shapes / dtypes runs the inner trace function exactly ONCE.
//
// The C++ helper increments a `static std::atomic<int>` inside the
// trace function. The first call (cache miss) drives `compile_trace`
// which invokes the trace fn → counter goes 0 → 1. The second call
// hits the compile cache via `CompilerCache::find` (keyed on shape +
// dtype + constants, NOT primitive identity), so the trace fn is
// NOT re-invoked → counter stays at 1.
//
// This is a pure-trace test — we never call `eval()` on the compiled
// outputs, so the GPU dispatch path never fires and the test passes
// on hosts without Metal.
// =============================================================================

#[test]
fn compile_trace_paged_kv_write_caches_one_trace() {
    // Reset to defend against earlier tests in the same process.
    unsafe { mlx_sys::mlx_paged_kv_write_trace_count_reset() };
    assert_eq!(
        unsafe { mlx_sys::mlx_paged_kv_write_trace_count_get() },
        0,
        "trace counter must be 0 after reset"
    );

    // The C++ helper now uses REAL data-backed arrays on both calls.
    // It calls the compiled function twice (different K/V values and
    // slot ranges), evals each call's outputs, and inspects the second
    // call's K-pool slots after eval. The return codes are:
    //   1   → success (one trace, second-call values found at second-
    //         call slots — both cache hit AND `compile_replace`'s
    //         runtime-thread is correct).
    //  -1   → internal/setup error.
    //  -2   → second-call slots did NOT contain second-call K values
    //         (compile_replace runtime-thread bug).
    //  -3   → Metal not available; eval-based verification skipped.
    //         Trace-count check still ran successfully (count == 1).
    //         Test passes as a no-op-success on this branch.
    let count = unsafe { mlx_sys::mlx_paged_kv_write_compile_trace_smoke(2) };

    if count == -3 {
        eprintln!(
            "compile_trace_paged_kv_write_caches_one_trace: \
             Metal not available; eval-based verification skipped"
        );
        // The trace-count assertion still ran inside the helper before
        // the Metal check; we only get -3 if `count_after_second == 1`.
        return;
    }

    assert_ne!(
        count, -2,
        "compile_replace runtime-thread bug: second-call slots did NOT \
         contain second-call K values (cache returned first-call inputs)"
    );
    assert_ne!(count, -1, "compile_trace helper hit an internal error");

    assert_eq!(
        count, 1,
        "expected exactly 1 trace across two same-shape calls (got {count}); \
         this means MLX's compile cache rejected our second call's shapes \
         (which would mean the cache key is unstable across re-runs)"
    );

    // Also verify the counter is observable from Rust.
    let observed = unsafe { mlx_sys::mlx_paged_kv_write_trace_count_get() };
    assert_eq!(observed, 1, "trace counter observable from Rust must agree");
}

// =============================================================================
// Phase 1 review-round-3 negative-validation tests.
//
// Each test calls a C++ helper that constructs `paged_attention` /
// `paged_kv_write` factory inputs that are well-formed EXCEPT for one
// specific dim or dtype, then asserts the factory throws
// `std::invalid_argument`. These guard the kernel buffer-contract
// requirements at the factory level so a malformed caller cannot
// tunnel out-of-bounds GPU reads through the dispatcher.
// =============================================================================

#[test]
fn paged_attention_factory_rejects_q_rank_not_3() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_q_rank_not_3() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject q with rank != 3 (got {threw})"
    );
}

#[test]
fn paged_attention_factory_rejects_block_table_batch_mismatch() {
    let threw =
        unsafe { mlx_sys::mlx_paged_attention_factory_rejects_block_table_batch_mismatch() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject block_table.shape(0) != q.shape(0) \
         (got {threw}); kernel addresses block_tables[seq_idx*max_blocks_per_seq], \
         a mismatch reads past the buffer"
    );
}

#[test]
fn paged_attention_factory_rejects_block_table_dtype() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_block_table_dtype() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject non-int32 block_table dtype (got {threw}); \
         kernel reinterprets the buffer as 32-bit indices"
    );
}

#[test]
fn paged_attention_factory_rejects_seq_lens_batch_mismatch() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_seq_lens_batch_mismatch() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject seq_lens.shape(0) != q.shape(0) (got {threw})"
    );
}

#[test]
fn paged_attention_factory_rejects_k_pool_inner_dim() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_k_pool_inner_dim() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject k_pool.shape(2) != head_size/x_pack (got {threw}); \
         kernel reads K[block, h, d/x, t, x] and a mismatched inner dim re-routes bytes"
    );
}

#[test]
fn paged_attention_factory_rejects_k_pool_x_pack() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_k_pool_x_pack() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject k_pool.shape(4) != dtype-derived x_pack (got {threw})"
    );
}

#[test]
fn paged_attention_factory_rejects_v_pool_head_dim() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_v_pool_head_dim() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject v_pool.shape(2) != head_size (got {threw})"
    );
}

#[test]
fn paged_attention_factory_rejects_num_blocks_mismatch() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_num_blocks_mismatch() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject k_pool.shape(0) != v_pool.shape(0) (got {threw})"
    );
}

#[test]
fn paged_kv_write_factory_rejects_slot_mapping_rank() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_slot_mapping_rank() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject slot_mapping with rank != 1 (got {threw})"
    );
}

#[test]
fn paged_kv_write_factory_rejects_slot_mapping_dtype() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_slot_mapping_dtype() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject slot_mapping with dtype != int64 (got {threw}); \
         kernel reads slot_mapping[token_idx] as `int64_t*`"
    );
}

#[test]
fn paged_kv_write_factory_rejects_slot_mapping_length() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_slot_mapping_length() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject slot_mapping length != new_k.shape(0) (got {threw}); \
         a short mapping reads past the buffer for tokens beyond its length"
    );
}

#[test]
fn paged_kv_write_factory_rejects_slot_mapping_out_of_range() {
    // Phase 1 safety check: slot value >= num_blocks * block_size is
    // out-of-pool and must be rejected at the factory. This requires
    // real data (the eval-based bounds check), so the helper builds
    // real BF16 / int64 arrays.
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_slot_mapping_out_of_range() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject slot_mapping max >= num_blocks*block_size (got {threw}); \
         a slot beyond the pool capacity writes past the K/V allocation. Phase 2 will move \
         this check kernel-side."
    );
}

/// Round-13 regression: assert the factory's slot_mapping bounds
/// `std::invalid_argument` carries the `[runtime]` marker.
///
/// The slot_mapping bounds check is a data-dependent runtime guard
/// (the value of `max(slot_mapping)` cannot be known structurally —
/// it requires real materialized data). The same property is also
/// guarded inside `PagedKVWrite::eval_gpu` (the compile-cached path),
/// where the throw is tagged `[runtime] PagedKVWrite::eval_gpu`.
///
/// The companion factory throw must use the same `[runtime]` prefix so
/// runtime-content guards stay uniformly distinguishable from
/// `[validator]`-tagged structural rejections, regardless of which path
/// caught the bad data. Without this regression test, a naive rebase
/// could drop the prefix from the factory side without breaking any
/// existing assertion (the previous test only checked the exception
/// class).
#[test]
fn paged_kv_write_factory_runtime_guard_marker() {
    let rc =
        unsafe { mlx_sys::mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_marker() };
    match rc {
        1 => {} // success: factory threw + message contained "[runtime]"
        0 => panic!(
            "paged_kv_write factory did NOT throw on slot_mapping max >= \
             num_blocks*block_size; runtime-content bounds guard regressed. \
             A regression here would let out-of-pool slot ids reach the \
             Metal kernel, writing past the K/V allocation."
        ),
        -2 => panic!(
            "paged_kv_write factory threw std::invalid_argument on \
             out-of-range slot_mapping but the message did NOT contain \
             '[runtime]' — the prefix on the factory-side runtime guard \
             regressed. The factory and the eval_gpu-side bounds check \
             must BOTH tag their throws with '[runtime]' so callers can \
             uniformly distinguish runtime-content guards from \
             '[validator]'-tagged structural rejections. See stderr for \
             the actual message."
        ),
        -1 => panic!(
            "paged_kv_write factory threw a non-invalid_argument exception \
             OR the helper hit a setup error before the factory could \
             throw. See stderr for the underlying message."
        ),
        other => panic!(
            "paged_kv_write_factory_runtime_guard_marker: helper returned \
             unexpected rc={other}; valid codes are 0 (no throw), 1 (pass), \
             -1 (internal error), -2 (marker missing)."
        ),
    }
}

// =============================================================================
// Phase 1 review-round-4 dtype-mismatch tests (finding B + C).
//
// The factory previously checked only pairwise dtype equality (k_pool ==
// v_pool, new_k == new_v). It did NOT verify the dtype matched the
// cache/io dtype implied by `kv_dtype`. Round 4 adds factory-side
// validation: any dtype slot that disagrees with `kv_dtype`'s expected
// (cache, io) pair must be rejected with `std::invalid_argument`.
// =============================================================================

#[test]
fn paged_kv_write_factory_rejects_k_pool_dtype_bf16() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_k_pool_dtype_bf16() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject k_pool dtype != bfloat16 for kv_dtype=Bf16 \
         (got {threw}); a dtype mismatch silently misroutes the Metal kernel template"
    );
}

#[test]
fn paged_kv_write_factory_rejects_v_pool_dtype_bf16() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_v_pool_dtype_bf16() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject v_pool dtype != bfloat16 for kv_dtype=Bf16 (got {threw})"
    );
}

#[test]
fn paged_kv_write_factory_rejects_new_k_dtype_bf16() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_new_k_dtype_bf16() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject new_k dtype != bfloat16 for kv_dtype=Bf16 (got {threw})"
    );
}

#[test]
fn paged_kv_write_factory_rejects_new_v_dtype_bf16() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_new_v_dtype_bf16() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject new_v dtype != bfloat16 for kv_dtype=Bf16 (got {threw})"
    );
}

#[test]
fn paged_kv_write_factory_rejects_k_pool_dtype_fp8() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_k_pool_dtype_fp8() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject k_pool dtype != uint8 for kv_dtype=Fp8 \
         (got {threw}); FP8 cache is stored opaquely as bytes"
    );
}

#[test]
fn paged_kv_write_factory_rejects_new_k_dtype_fp8() {
    let threw = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_new_k_dtype_fp8() };
    assert_eq!(
        threw, 1,
        "paged_kv_write(...) must reject new_k dtype != bfloat16 for kv_dtype=Fp8 \
         (got {threw}); Phase 1 contract: FP8 io dtype is bfloat16"
    );
}

#[test]
fn paged_attention_factory_rejects_q_dtype_bf16() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_q_dtype_bf16() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject q dtype != bfloat16 for kv_dtype=Bf16 (got {threw})"
    );
}

#[test]
fn paged_attention_factory_rejects_q_dtype_fp8() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_q_dtype_fp8() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject q dtype != bfloat16 for kv_dtype=Fp8 \
         (got {threw}); Phase 1 contract: FP8 io dtype is bfloat16"
    );
}

#[test]
fn paged_attention_factory_rejects_k_pool_dtype_bf16() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_k_pool_dtype_bf16() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject k_pool dtype != bfloat16 for kv_dtype=Bf16 (got {threw})"
    );
}

#[test]
fn paged_attention_factory_rejects_k_pool_dtype_fp8() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_k_pool_dtype_fp8() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject k_pool dtype != uint8 for kv_dtype=Fp8 (got {threw})"
    );
}

// =============================================================================
// Phase 1 review-round-6 finding: GQA head-group divisibility.
//
// The Metal kernel computes:
//   num_queries_per_kv = num_heads / num_kv_heads
//   kv_head_idx        = head_idx / num_queries_per_kv
// (paged_attention.metal:839-840). A malformed (num_q_heads,
// num_kv_heads) pair turns a structurally shape-consistent call into
// a GPU fault (division by zero) or an out-of-pool K/V read. The
// factory now rejects three invariant violations up front:
//   1. num_kv_heads == 0
//   2. num_q_heads <  num_kv_heads (q-per-kv is 0; div-by-zero risk)
//   3. num_q_heads % num_kv_heads != 0 (later heads index past KV dim)
// =============================================================================

#[test]
fn paged_attention_factory_rejects_zero_kv_heads() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_zero_kv_heads() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject num_kv_heads=0 (got {threw}); \
         the kernel computes num_queries_per_kv = num_heads / num_kv_heads, \
         which would divide by zero."
    );
}

#[test]
fn paged_attention_factory_rejects_q_heads_less_than_kv_heads() {
    let threw =
        unsafe { mlx_sys::mlx_paged_attention_factory_rejects_q_heads_less_than_kv_heads() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject num_q_heads (2) < num_kv_heads (4) \
         (got {threw}); the kernel would compute num_queries_per_kv = 2/4 = 0 \
         (integer division) and then divide head_idx by zero on the kv_head_idx \
         line."
    );
}

#[test]
fn paged_attention_factory_rejects_indivisible_grouping() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_indivisible_grouping() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject num_q_heads (6) % num_kv_heads (4) != 0 \
         (got {threw}); 6 / 4 = 1, so head_idx = 4, 5 would compute kv_head_idx \
         = 4, 5 — past the end of the 4-entry KV-head pool dimension."
    );
}

// =============================================================================
// Phase 1 review-round-7 finding: PagedAttention factory must reject
// `block_size == 0` and `head_size == 0`.
//
// The pool-shape equality check only verifies `pool.shape(3) ==
// block_size`, so a zero-sized pool block dimension passes when
// `block_size=0`. On `eval_gpu`, the runtime bounds check then computes
// `(s + block_size - 1) / block_size` and divides by zero in host code
// BEFORE the later `max_context_len <= 0` guard could reject — a
// process-crash hazard. `head_size == 0` is the symmetric case and is
// already rejected in `paged_kv_write`'s factory; mirror it here.
// =============================================================================

#[test]
fn paged_attention_factory_rejects_zero_block_size() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_zero_block_size() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject block_size = 0 (got {threw}); \
         the pool shape equality check accepts a zero-sized pool block \
         dim when block_size=0, and `eval_gpu`'s subsequent bounds check \
         would then divide by zero in `(s + block_size - 1) / block_size`."
    );
}

#[test]
fn paged_attention_factory_rejects_zero_head_size() {
    let threw = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_zero_head_size() };
    assert_eq!(
        threw, 1,
        "paged_attention(...) must reject head_size = 0 (got {threw}); \
         the Metal kernel uses head_size as a grid extent and indexing \
         stride. Mirrors `paged_kv_write`'s identical check for symmetry."
    );
}

// =============================================================================
// Phase 1 review-round-4 finding A: compile-cached path slot-bounds test.
//
// The factory's slot bounds check is skipped during MLX tracing AND on
// compile-cache hits (the cache routes runtime `slot_mapping` straight
// into `eval_gpu`). The fix moves the bounds check into
// `PagedKVWrite::eval_gpu` so it fires on EVERY runtime call. This test
// verifies the runtime check triggers when the cached graph's eval is
// invoked with an out-of-range slot.
// =============================================================================

#[test]
fn compile_cached_paged_kv_write_oob_slot_throws() {
    // The C++ helper:
    //   1. Compiles a function emitting `paged_kv_write`.
    //   2. Calls it with valid slot_mapping = [0, 16] — cache miss
    //      triggers the factory's eval-based check (passes).
    //   3. Calls it again with the SAME shapes but
    //      slot_mapping = [0, num_blocks * block_size = 64] —
    //      out-of-range. Cache HIT bypasses the factory.
    //      `PagedKVWrite::eval_gpu`'s own bounds check MUST throw.
    //
    // Return codes:
    //   1   → success (eval_gpu threw on the out-of-range slot).
    //   0   → regression (eval_gpu did NOT throw — the kernel would
    //         have written past the K/V pool).
    //  -1   → internal/setup error (first call failed).
    //  -3   → Metal not available; verification skipped.
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_compile_cached_oob_throws() };

    if rc == -3 {
        eprintln!(
            "compile_cached_paged_kv_write_oob_slot_throws: \
             Metal not available; skipping eval-based verification"
        );
        return;
    }

    assert_ne!(rc, -1, "compile-cached OOB helper hit an internal error");
    assert_eq!(
        rc, 1,
        "PagedKVWrite::eval_gpu must throw std::invalid_argument when the \
         compile-cached path receives a slot_mapping with max >= num_blocks * \
         block_size (got rc={rc}). The factory check is skipped on cache hits, \
         so the bounds check must fire kernel-side / eval_gpu-side. Without it, \
         the Metal kernel writes past the K/V pool."
    );
}

// =============================================================================
// Phase 1 review-round-5: PagedAttention::eval_gpu runtime-content checks.
//
// The PagedAttention factory only validates structural (rank/shape/dtype)
// invariants. Without per-call validation in `eval_gpu`, the Metal
// kernel reinterprets a negative `seq_lens[i]` as a huge unsigned
// context length, reads past the row's block-table region for a
// too-large `seq_lens[i]`, and dereferences arbitrary GPU memory for
// a `block_table[i,j]` outside `[0, num_blocks)`. The fix puts the
// runtime check in `PagedAttention::eval_gpu`, so it fires on every
// replay (factory-direct and compile-cached). These tests verify the
// check via real-data eval.
// =============================================================================

/// Helper: interpret a return code from a `paged_attention` eval_gpu
/// helper. `rc==-3` means Metal unavailable (skip with stderr note);
/// `rc==1` means the expected throw fired.
fn assert_paged_attention_eval_gpu_rejects(rc: i32, scenario: &str) {
    if rc == -3 {
        eprintln!("paged_attention eval_gpu rejection ({scenario}): Metal not available; skipping");
        return;
    }
    assert_ne!(rc, -1, "{scenario}: helper hit an internal error (rc=-1)");
    assert_eq!(
        rc, 1,
        "{scenario}: PagedAttention::eval_gpu must throw std::invalid_argument \
         (got rc={rc}). Without the runtime bounds check the Metal kernel \
         either reads past the row's block-table region or dereferences \
         out-of-pool K/V memory."
    );
}

#[test]
fn paged_attention_eval_gpu_rejects_negative_seq_len() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_negative_seq_len() };
    assert_paged_attention_eval_gpu_rejects(rc, "negative seq_lens");
}

#[test]
fn paged_attention_eval_gpu_rejects_oversized_seq_len() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_oversized_seq_len() };
    assert_paged_attention_eval_gpu_rejects(rc, "seq_lens > max_blocks_per_seq*block_size");
}

#[test]
fn paged_attention_eval_gpu_rejects_negative_block_id() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_negative_block_id() };
    assert_paged_attention_eval_gpu_rejects(rc, "negative block_table entry");
}

#[test]
fn paged_attention_eval_gpu_rejects_oob_block_id() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_oob_block_id() };
    assert_paged_attention_eval_gpu_rejects(rc, "block_table entry >= num_blocks");
}

#[test]
fn compile_cached_paged_attention_oob_block_throws() {
    // The C++ helper:
    //   1. Compiles a function emitting `paged_attention`.
    //   2. Calls it with valid block_table = [0, 0, 0, 0],
    //      seq_lens = [16] — cache miss; factory + eval_gpu both pass.
    //   3. Calls it again with the SAME shapes but
    //      block_table[0, 0] = num_blocks = 4 — out of range. Cache HIT
    //      bypasses the factory. `PagedAttention::eval_gpu`'s own
    //      bounds check MUST throw `std::invalid_argument`.
    //
    // Return codes:
    //   1   → success (eval_gpu threw on the OOB block id).
    //   0   → regression (eval_gpu did NOT throw — the kernel would
    //         have addressed out-of-pool K/V memory).
    //  -1   → internal/setup error (first call failed).
    //  -3   → Metal not available; verification skipped.
    let rc = unsafe { mlx_sys::mlx_paged_attention_compile_cached_oob_throws() };

    if rc == -3 {
        eprintln!(
            "compile_cached_paged_attention_oob_block_throws: \
             Metal not available; skipping eval-based verification"
        );
        return;
    }

    assert_ne!(
        rc, -1,
        "compile-cached paged_attention OOB helper hit an internal error"
    );
    assert_eq!(
        rc, 1,
        "PagedAttention::eval_gpu must throw std::invalid_argument when the \
         compile-cached path receives a block_table with an entry >= num_blocks \
         (got rc={rc}). The factory check covers only structural shape/dtype, \
         and `mlx::core::compile`'s cached re-traces bypass the factory entirely. \
         Without the eval_gpu-side check the kernel would dereference \
         out-of-pool K/V memory."
    );
}

// =============================================================================
// Phase 1 review-round-8: factory must reject non-row-contiguous /
// nonzero-offset views for ALL inputs.
//
// The dispatch path forwards raw MTLBuffer pointers + at most a single
// offset (q/out for paged_attention; slot_mapping/new_k/new_v for
// paged_kv_write). Pool-side and metadata-side inputs (k_pool, v_pool,
// block_table, seq_lens) are passed as bare buffers with no offset,
// and strides are never forwarded for any input. A sliced/transposed
// view with a valid logical shape would silently alias the wrong
// region of the backing allocation.
//
// The factory now requires every input to satisfy
// `flags().row_contiguous == true && offset() == 0`. Tests below
// confirm rejection fires for transposed q / sliced k_pool / sliced
// block_table / sliced seq_lens. Each test is gated on Metal
// availability (slice/transpose eval requires it) and skips with a
// log message when unavailable.
// =============================================================================

#[test]
fn paged_kv_write_factory_rejects_non_contiguous_k_pool() {
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_factory_rejects_non_contiguous_k_pool() };
    if rc == -3 {
        eprintln!(
            "paged_kv_write_factory_rejects_non_contiguous_k_pool: \
             Metal not available; skipping"
        );
        return;
    }
    assert_eq!(
        rc, 1,
        "paged_kv_write(...) must reject a sliced (non-row-contiguous, \
         nonzero-offset) k_pool view (got rc={rc}); the dispatch passes \
         k_pool as a bare MTLBuffer with no offset, so a sliced view \
         would clobber unrelated regions of the backing allocation."
    );
}

#[test]
fn paged_attention_factory_rejects_non_contiguous_q() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_non_contiguous_q() };
    if rc == -3 {
        eprintln!(
            "paged_attention_factory_rejects_non_contiguous_q: \
             Metal not available; skipping"
        );
        return;
    }
    assert_eq!(
        rc, 1,
        "paged_attention(...) must reject a transposed (non-row-contiguous) \
         q (got rc={rc}); the kernel reads q as a dense row-major buffer \
         and a transposed view's strides would not match its logical shape."
    );
}

#[test]
fn paged_attention_factory_rejects_non_contiguous_block_table() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_non_contiguous_block_table() };
    if rc == -3 {
        eprintln!(
            "paged_attention_factory_rejects_non_contiguous_block_table: \
             Metal not available; skipping"
        );
        return;
    }
    assert_eq!(
        rc, 1,
        "paged_attention(...) must reject a sliced block_table view (got rc={rc}); \
         the dispatch passes block_table as a bare MTLBuffer with no offset, \
         so a sliced view would read from offset 0 instead of the logical \
         slice start."
    );
}

#[test]
fn paged_attention_factory_rejects_non_contiguous_seq_lens() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_factory_rejects_non_contiguous_seq_lens() };
    if rc == -3 {
        eprintln!(
            "paged_attention_factory_rejects_non_contiguous_seq_lens: \
             Metal not available; skipping"
        );
        return;
    }
    assert_eq!(
        rc, 1,
        "paged_attention(...) must reject a sliced seq_lens view (got rc={rc}); \
         the dispatch passes seq_lens as a bare MTLBuffer with no offset, \
         so a `seq_lens[1:]` view would read from offset 0 instead of \
         the logical slice start."
    );
}

#[test]
fn paged_attention_forward_rejects_non_contiguous_metadata() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_forward_rejects_non_contiguous_metadata() };
    if rc == -3 {
        eprintln!(
            "paged_attention_forward_rejects_non_contiguous_metadata: \
             Metal not available; skipping"
        );
        return;
    }
    assert_eq!(
        rc, 1,
        "mlx_paged_attention_forward must reject sliced metadata (got rc={rc}); \
         the bridge must not mask block_table/seq_lens views with lazy \
         contiguous copies because eval_gpu reads metadata host-side."
    );
}

#[test]
fn paged_attention_forward_accepts_materialized_metadata() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_forward_eval_accepts_materialized_metadata() };
    if rc == -3 {
        eprintln!(
            "paged_attention_forward_accepts_materialized_metadata: \
             Metal not available; skipping"
        );
        return;
    }
    assert_ne!(
        rc, -1,
        "mlx_paged_attention_forward valid-metadata helper hit an internal error"
    );
    assert_eq!(
        rc, 1,
        "mlx_paged_attention_forward must accept materialized, row-contiguous \
         block_table/seq_lens and evaluate the lazy output successfully \
         (got rc={rc})."
    );
}

#[test]
fn paged_kv_write_forward_accepts_materialized_metadata() {
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_forward_eval_smoke() };
    if rc == -3 {
        eprintln!(
            "paged_kv_write_forward_accepts_materialized_metadata: \
             Metal not available; skipping"
        );
        return;
    }
    assert_ne!(
        rc, -1,
        "mlx_paged_kv_write_forward smoke hit an eval-time error"
    );
    assert_eq!(
        rc, 1,
        "mlx_paged_kv_write_forward must accept materialized slot_mapping \
         metadata and evaluate the lazy pool outputs successfully (got rc={rc})."
    );
}

// =============================================================================
// Phase 1 review-round-9: compile-cached eval_gpu must mirror the
// factory's row-contiguous / zero-offset check.
//
// `mlx::core::compile` cache hits rebuild the cached primitive with real
// inputs via `compile_replace` WITHOUT re-running the factory — the
// cache key only compares rank/shape/dtype. A graph first traced with
// contiguous inputs can later be replayed with a same-shape
// sliced/transposed view that bypasses the factory's check entirely.
// The fix mirrors `require_row_contiguous_zero_offset` inside both
// `PagedKVWrite::eval_gpu` and `PagedAttention::eval_gpu`. These tests
// verify the mirrored check fires on the compile-cached second eval.
// =============================================================================

#[test]
fn compile_cached_paged_kv_write_rejects_non_contiguous() {
    // The C++ helper:
    //   1. Compiles a function emitting `paged_kv_write`.
    //   2. Calls it with contiguous inputs (cache miss; factory +
    //      eval_gpu both pass).
    //   3. Calls it again with the SAME shapes/dtypes but with `new_k`
    //      substituted by a transposed (non-row-contiguous) view.
    //      Cache HIT bypasses the factory. The mirrored
    //      `require_row_contiguous_zero_offset` check inside
    //      `PagedKVWrite::eval_gpu` MUST throw `std::invalid_argument`.
    //
    // Return codes:
    //   1   → success (eval_gpu threw on the non-contiguous view).
    //   0   → regression (eval_gpu did NOT throw — the kernel would
    //         have aliased the wrong region of the new_k buffer).
    //  -1   → internal/setup error (first call failed).
    //  -3   → Metal not available; verification skipped (slice/
    //         transpose materialization needs Metal).
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_compile_cached_non_contiguous_throws() };

    if rc == -3 {
        eprintln!(
            "compile_cached_paged_kv_write_rejects_non_contiguous: \
             Metal not available; skipping eval-based verification"
        );
        return;
    }

    assert_ne!(
        rc, -1,
        "compile-cached paged_kv_write non-contiguous helper hit an internal error"
    );
    assert_eq!(
        rc, 1,
        "PagedKVWrite::eval_gpu must throw std::invalid_argument when the \
         compile-cached path receives a non-row-contiguous input view \
         (got rc={rc}). The factory's contiguity check is skipped on \
         cache hits, so the mirrored check must fire eval_gpu-side. \
         Without it, the dispatch would silently alias the wrong region \
         of the backing allocation."
    );
}

#[test]
fn compile_cached_paged_attention_rejects_non_contiguous() {
    // The C++ helper:
    //   1. Compiles a function emitting `paged_attention`.
    //   2. Calls it with contiguous inputs (cache miss; factory +
    //      eval_gpu both pass).
    //   3. Calls it again with the SAME shapes/dtypes but with
    //      `block_table` substituted by a sliced (nonzero-offset) view
    //      of a wider [3, 4] backing buffer (rows [1:2] → shape [1, 4]
    //      at nonzero offset). Cache HIT bypasses the factory. The
    //      mirrored `require_row_contiguous_zero_offset` check inside
    //      `PagedAttention::eval_gpu` MUST throw `std::invalid_argument`.
    //
    // Return codes:
    //   1   → success (eval_gpu threw on the non-contiguous view).
    //   0   → regression (eval_gpu did NOT throw — the kernel would
    //         have read from offset 0 of the backing buffer instead of
    //         the logical slice start).
    //  -1   → internal/setup error (first call failed).
    //  -3   → Metal not available; verification skipped (slice
    //         materialization needs Metal).
    let rc = unsafe { mlx_sys::mlx_paged_attention_compile_cached_non_contiguous_throws() };

    if rc == -3 {
        eprintln!(
            "compile_cached_paged_attention_rejects_non_contiguous: \
             Metal not available; skipping eval-based verification"
        );
        return;
    }

    assert_ne!(
        rc, -1,
        "compile-cached paged_attention non-contiguous helper hit an internal error"
    );
    assert_eq!(
        rc, 1,
        "PagedAttention::eval_gpu must throw std::invalid_argument when the \
         compile-cached path receives a non-row-contiguous / nonzero-offset \
         input view (got rc={rc}). The factory's contiguity check is skipped \
         on cache hits, so the mirrored check must fire eval_gpu-side. \
         Without it, the dispatch would silently read from the wrong region \
         of the backing allocation."
    );
}

// =============================================================================
// Phase 1 review-round-10 finding: factory + eval_gpu validation must be
// unified via a shared helper.
//
// Rounds 3-9 each surfaced another factory-only check that
// `mlx::core::compile`'s cache-hit replay bypassed (compile_replace
// rebuilds the cached primitive with real inputs WITHOUT re-running
// the factory). Round 10 refactors the entire structural / scalar /
// shape / dtype / contiguity validation into a shared helper that the
// factory AND eval_gpu BOTH call, closing the entire class of hazards
// in one swoop.
//
// These tests prove eval_gpu now catches bad scalar state on its own
// (without relying on the factory having already rejected the inputs).
// The C++ helper directly constructs a primitive with deliberately
// bad scalar state, wires it into an MLX graph via `array::make_arrays`,
// and `eval()`s the result. The factory is never invoked, so the throw
// must come from the validator inside `eval_gpu` itself.
// =============================================================================

/// Assert the C++ helper proved that **`eval_gpu`** (not the
/// graph-construction step) is the throwing site. Round-11 tightening:
/// the helper now distinguishes graph-construction rejection from
/// eval_gpu rejection so a pre-eval throw can no longer masquerade as
/// success. Return codes:
///
/// - `1` — eval threw `std::invalid_argument` AND the message contains
///   the eval_gpu validator context. **Pass.**
/// - `0` — eval did not throw at all. Bad scalar state was silently
///   accepted. **Fail.**
/// - `2` — graph construction threw `std::invalid_argument` BEFORE eval
///   ran. The helper's structurally-valid inputs should never trigger
///   this — internal helper bug. **Fail.**
/// - `-1` — non-`std::invalid_argument` exception in either step.
///   **Fail.**
/// - `-2` — eval threw `std::invalid_argument` but the message did not
///   contain the eval_gpu validator context. The throw came from
///   somewhere other than the validator we are exercising. **Fail.**
fn assert_eval_gpu_rejects_bad_state(rc: i32, scenario: &str) {
    match rc {
        1 => {} // success
        0 => panic!(
            "{scenario}: eval_gpu DID NOT throw — bad scalar state was \
             silently accepted. Without the validator inside eval_gpu, a \
             compile-cache replay (which bypasses the factory) could route \
             a primitive with bad scalar state to the dispatch path."
        ),
        2 => panic!(
            "{scenario}: INTERNAL HELPER BUG — graph construction \
             (make_arrays / array constructor) threw \
             std::invalid_argument BEFORE eval ran. This means the \
             helper is not actually exercising eval_gpu's validator; \
             the rejection is happening at the wrong layer. Fix the \
             helper to keep its structural inputs valid so the bad \
             scalar state must be caught by eval_gpu, not by graph \
             construction."
        ),
        -1 => panic!(
            "{scenario}: helper hit an internal error (rc=-1) — a \
             non-invalid_argument exception fired in either \
             construction or eval. See stderr for the underlying \
             message."
        ),
        -2 => panic!(
            "{scenario}: eval threw std::invalid_argument but the \
             message did not satisfy BOTH required markers — the \
             validator-only token \"[validator]\" AND the operation \
             tag (e.g. \"PagedKVWrite::eval_gpu\" / \
             \"PagedAttention::eval_gpu\"). The throw is coming from \
             either a different layer of the eval path (no op tag) or \
             from a runtime-content guard inside eval_gpu (op tag \
             present but no \"[validator]\" marker; those guards use \
             \"[runtime] ...\"). Either failure mode means the scalar \
             validator regressed and a non-validator throw is masking \
             it. See stderr for the actual message."
        ),
        other => panic!(
            "{scenario}: helper returned unexpected rc={other}; valid \
             codes are 0 (eval did not throw), 1 (eval_gpu validator \
             threw — pass), 2 (graph construction threw — internal \
             helper bug), -1 (non-invalid_argument), -2 (wrong throw \
             site)."
        ),
    }
}

#[test]
fn paged_kv_write_eval_gpu_rejects_zero_kv_heads() {
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_eval_gpu_rejects_zero_kv_heads() };
    assert_eval_gpu_rejects_bad_state(rc, "PagedKVWrite num_kv_heads=0");
}

#[test]
fn paged_kv_write_eval_gpu_rejects_zero_block_size() {
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_eval_gpu_rejects_zero_block_size() };
    assert_eval_gpu_rejects_bad_state(rc, "PagedKVWrite block_size=0");
}

/// Round-12 regression: prove the scalar validator (NOT the runtime
/// slot_mapping bounds guard) is the throw site for `block_size=0`.
///
/// The companion `..._rejects_zero_block_size` test uses
/// `slot_mapping={0,16}`, which means a regressed scalar validator
/// would let the runtime guard fire on `max_slot=16 >= pool_capacity=
/// num_blocks*block_size=0` — and a substring-only check on
/// "PagedKVWrite::eval_gpu" would silently report rc=1 even though
/// the validator is gone.
///
/// This regression test deliberately uses `slot_mapping={-1,-1}`
/// (all "skip" sentinels). The runtime guard explicitly excludes
/// negative slot ids from its max-slot reduction
/// (`if (slot_data[i] >= 0 ...)` in PagedKVWrite::eval_gpu), so the
/// guard physically cannot throw on this input. The ONLY remaining
/// throw site capable of producing `std::invalid_argument` is the
/// scalar validator's `block_size <= 0` reject. If that reject
/// regressed, the helper would proceed all the way to the metal
/// dispatch (returning rc=0 — eval did not throw). If the
/// `[validator]` marker were stripped, rc=-2 would surface.
#[test]
fn paged_kv_write_eval_gpu_validator_proof_zero_block_size() {
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_eval_gpu_validator_proof_zero_block_size() };
    assert_eval_gpu_rejects_bad_state(
        rc,
        "PagedKVWrite block_size=0 with benign slot_mapping (proves \
         the scalar validator, not the runtime guard, is the throw \
         site — runtime guard is excluded by sentinel slot ids)",
    );
}

#[test]
fn paged_kv_write_eval_gpu_rejects_zero_head_size() {
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_eval_gpu_rejects_zero_head_size() };
    assert_eval_gpu_rejects_bad_state(rc, "PagedKVWrite head_size=0");
}

#[test]
fn paged_kv_write_eval_gpu_rejects_x_pack_dtype_mismatch() {
    let rc = unsafe { mlx_sys::mlx_paged_kv_write_eval_gpu_rejects_x_pack_dtype_mismatch() };
    assert_eval_gpu_rejects_bad_state(rc, "PagedKVWrite x_pack=16 vs Bf16 (expects x_pack=8)");
}

#[test]
fn paged_attention_eval_gpu_rejects_zero_kv_heads() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_zero_kv_heads() };
    assert_eval_gpu_rejects_bad_state(rc, "PagedAttention num_kv_heads=0");
}

#[test]
fn paged_attention_eval_gpu_rejects_indivisible_grouping() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_indivisible_grouping() };
    assert_eval_gpu_rejects_bad_state(
        rc,
        "PagedAttention num_q_heads=6 not divisible by num_kv_heads=4",
    );
}

#[test]
fn paged_attention_eval_gpu_rejects_sliding_window() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_sliding_window() };
    assert_eval_gpu_rejects_bad_state(
        rc,
        "PagedAttention sliding_window=-1 (negative values illegal; \
         Phase 7 enables sliding_window > 0)",
    );
}

#[test]
fn paged_attention_eval_gpu_rejects_zero_block_size() {
    let rc = unsafe { mlx_sys::mlx_paged_attention_eval_gpu_rejects_zero_block_size() };
    assert_eval_gpu_rejects_bad_state(rc, "PagedAttention block_size=0");
}
