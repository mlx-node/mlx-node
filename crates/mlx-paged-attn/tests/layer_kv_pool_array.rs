//! Integration test for `LayerKVPool::{key,value}_cache_array_raw`.
//!
//! Verifies the zero-copy MLX-array view shape, dtype, and round-trip
//! behaviour against a real `LayerKVPool` allocation. Skipped on no-Metal
//! hosts (the underlying buffer allocation requires a Metal device).
//!
//! These tests exercise the FFI path:
//! `mlx_array_from_metal_buffer_view(MTL::Buffer*, dims, ndim, dtype)`,
//! the same FFI that the higher-level `PagedKVCacheAdapter::{key,value}_
//! pool_array` thin wrappers in `mlx-core` call.

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};

fn cfg(num_layers: u32) -> PagedAttentionConfig {
    PagedAttentionConfig {
        block_size: 8,
        gpu_memory_mb: 256,
        head_size: 64,
        num_kv_heads: 2,
        num_layers,
        use_fp8_cache: Some(false),
        max_seq_len: Some(64),
        max_batch_size: Some(2),
    }
}

/// Skip helper: returns `None` on no-Metal hosts so each test can early-
/// return without panicking. Pool allocation needs `MetalState::get()`,
/// which fails on CI runners without a GPU.
fn maybe_pool(num_blocks: u32, dtype: MetalDtype) -> Option<LayerKVPool> {
    match LayerKVPool::new(cfg(2), num_blocks, dtype) {
        Ok(p) => Some(p),
        Err(e) if e.contains("No Metal device found") => None,
        Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
    }
}

/// BF16 pool: K view shape is `[num_blocks, num_kv_heads, head_size/x,
/// block_size, x]` with x=8. V view shape is `[num_blocks, num_kv_heads,
/// head_size, block_size]`. Round-trip confirms the pointer is non-null
/// for every layer and `mlx_array_delete` cleans up without releasing
/// the buffer (the pool stays usable afterwards).
#[test]
fn bf16_view_shapes_and_round_trip() {
    let Some(pool) = maybe_pool(4, MetalDtype::BFloat16) else {
        eprintln!("skipping bf16_view_shapes_and_round_trip: Metal unavailable");
        return;
    };

    // Hand-checked shape per cfg: num_blocks=4, num_kv_heads=2,
    // head_size=64, block_size=8, x=8 → K=[4,2,8,8,8], V=[4,2,64,8].
    let x = pool.cache_pack_factor().expect("pack factor");
    assert_eq!(x, 8);
    assert_eq!(pool.key_cache_shape(x), [4, 2, 8, 8, 8]);
    assert_eq!(pool.value_cache_shape(), [4, 2, 64, 8]);

    for layer_idx in 0..pool.num_layers() as u32 {
        let k = pool.key_cache_array_raw(layer_idx).expect("k view");
        assert!(
            !k.is_null(),
            "K view must be non-null for layer {layer_idx}"
        );
        // SAFETY: handle is non-null and from `mlx_array_from_metal_buffer_view`,
        // which heap-allocates via `new mlx::core::array(...)` — its destructor
        // calls our no-op deleter on the buffer (i.e. the buffer survives).
        unsafe { mlx_sys::mlx_array_delete(k) };

        let v = pool.value_cache_array_raw(layer_idx).expect("v view");
        assert!(
            !v.is_null(),
            "V view must be non-null for layer {layer_idx}"
        );
        unsafe { mlx_sys::mlx_array_delete(v) };
    }

    // Round-tripping the views does not invalidate the pool — we can
    // immediately request another view.
    let k2 = pool.key_cache_array_raw(0).expect("re-take view");
    assert!(
        !k2.is_null(),
        "pool must remain usable after view round-trip"
    );
    unsafe { mlx_sys::mlx_array_delete(k2) };
}

/// FP8 pool: x=16, V layout unchanged. Pins the FP8 path through the
/// same FFI.
#[test]
fn fp8_view_shapes() {
    let cfg = PagedAttentionConfig {
        block_size: 16,
        use_fp8_cache: Some(true),
        ..cfg(2)
    };
    let pool = match LayerKVPool::new(cfg, 4, MetalDtype::UChar) {
        Ok(p) => p,
        Err(e) if e.contains("No Metal device found") => {
            eprintln!("skipping fp8_view_shapes: Metal unavailable");
            return;
        }
        Err(e) => panic!("unexpected new failure: {e}"),
    };
    assert_eq!(pool.cache_pack_factor().unwrap(), 16);
    assert_eq!(pool.key_cache_shape(16), [4, 2, 4, 16, 16]);
    assert_eq!(pool.value_cache_shape(), [4, 2, 64, 16]);

    let k = pool.key_cache_array_raw(0).expect("k view fp8");
    assert!(!k.is_null());
    unsafe { mlx_sys::mlx_array_delete(k) };
}

/// Out-of-range layer index returns a clear error, not a panic.
#[test]
fn out_of_range_layer_errors() {
    let Some(pool) = maybe_pool(4, MetalDtype::BFloat16) else {
        eprintln!("skipping out_of_range_layer_errors: Metal unavailable");
        return;
    };

    let res = pool.key_cache_array_raw(99);
    assert!(res.is_err(), "out-of-range layer must return Err");
    let msg = res.unwrap_err();
    assert!(
        msg.contains("layer_idx") && msg.contains("out of range"),
        "expected layer-range error, got: {msg}"
    );
}

/// LIFETIME: the FFI helper `mlx_array_from_metal_buffer_view` retains
/// the underlying `MTL::Buffer*` and the array's deleter releases it on
/// drop, so the array view is INDEPENDENT of the original `metal::Buffer`
/// holder lifetime. A no-op deleter would instead leave the array
/// pointing at a freed Metal buffer (GPU use-after-free) once the pool
/// drops.
///
/// This test takes a view, drops the pool, evaluates the view by
/// running an MLX op against it (`astype` + `eval` materializes the
/// data through the lazy graph), and asserts no crash. If the buffer
/// were freed when the pool dropped, this would either segfault or
/// surface a Metal validation-layer error.
#[test]
fn buffer_view_outlives_pool_drop() {
    let Some(pool) = maybe_pool(4, MetalDtype::BFloat16) else {
        eprintln!("skipping buffer_view_outlives_pool_drop: Metal unavailable");
        return;
    };

    // Take a view, then DROP the pool while keeping the view alive.
    let view_raw = pool.key_cache_array_raw(0).expect("k view");
    assert!(!view_raw.is_null());

    drop(pool);

    // The view must still be usable. We force an eval so MLX actually
    // dereferences the underlying Metal buffer; if the buffer were
    // released by dropping the pool, this would crash. The astype
    // (BF16 -> F32) is the simplest op that materializes the data
    // without depending on a specific shape.
    //
    // SAFETY: `view_raw` is a valid mlx_array* from
    // `mlx_array_from_metal_buffer_view`. We delete it after eval.
    unsafe {
        let recast = mlx_sys::mlx_array_astype(view_raw, /* float32 */ 0);
        assert!(
            !recast.is_null(),
            "astype on view post-pool-drop must not return null"
        );
        mlx_sys::mlx_array_eval(recast);
        // No crash and `eval` returns successfully → buffer survived
        // the pool drop.
        mlx_sys::mlx_array_delete(recast);
        mlx_sys::mlx_array_delete(view_raw);
    }
}

/// Multiple views over the same buffer all hold independent refcount
/// entries on the MTL::Buffer. Dropping the views in any order must
/// not crash and must not leave a dangling reference. We drop the
/// pool first, then walk through several views and free them in
/// reverse order so the last view to drop is responsible for the
/// final `release()` (the hardest case for the lifetime contract).
#[test]
fn multiple_views_independent_refcounts() {
    let Some(pool) = maybe_pool(4, MetalDtype::BFloat16) else {
        eprintln!("skipping multiple_views_independent_refcounts: Metal unavailable");
        return;
    };

    // Take 3 views over the same buffer.
    let v1 = pool.key_cache_array_raw(0).expect("v1");
    let v2 = pool.key_cache_array_raw(0).expect("v2");
    let v3 = pool.value_cache_array_raw(1).expect("v3");
    assert!(!v1.is_null() && !v2.is_null() && !v3.is_null());

    drop(pool);

    // Free in REVERSE order — the last view freed is the one that
    // drops the final Metal-buffer reference. If any of the
    // intermediate frees double-released, we'd crash here.
    unsafe {
        mlx_sys::mlx_array_delete(v2);
        mlx_sys::mlx_array_delete(v3);
        mlx_sys::mlx_array_delete(v1);
    }
    // Reaching here without crashing is the assertion.
}
