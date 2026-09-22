#![cfg(target_os = "macos")]

//! Executes the non-NAX kernel on any Apple GPU. Explicit masks provide an
//! independent stock-MLX reference; no large checkpoint is needed.
use mlx_core::array::mask::create_causal_mask;
use mlx_core::array::{
    DType, MxArray, scaled_dot_product_attention, scaled_dot_product_attention_causal,
    synchronize_and_clear_cache,
};
use mlx_core::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig};
use std::sync::{Arc, Mutex};

fn paged_continuation() {
    // Exercise production pool writes/gathers with an unaligned prefix and
    // nonconstant values, so wrong slots or stale write ordering cannot pass.
    let (context, query, blocks) = (269_u32, 33_u32, 17_u32);
    let config = PagedAttentionConfig {
        block_size: 16,
        gpu_memory_mb: 256,
        head_size: 256,
        num_kv_heads: 4,
        num_layers: 1,
        use_fp8_cache: Some(false),
        max_seq_len: Some(512),
        max_batch_size: Some(1),
    };
    let pool = Arc::new(LayerKVPool::new(config, blocks, blocks, MetalDtype::BFloat16).unwrap());
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(blocks, blocks, 16)));
    let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 16).unwrap();
    adapter.reset_for_new_request(1).unwrap();
    adapter.allocate_suffix_blocks(context).unwrap();
    let k =
        MxArray::random_normal(&[context as i64, 4, 256], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
    let v =
        MxArray::random_normal(&[context as i64, 4, 256], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
    for (start, end) in [(0_i64, 236_i64), (236, 269)] {
        adapter
            .record_tokens(&vec![1; (end - start) as usize])
            .unwrap();
        adapter
            .update_keys_values_native(
                0,
                &k.slice(&[start, 0, 0], &[end, 4, 256]).unwrap(),
                &v.slice(&[start, 0, 0], &[end, 4, 256]).unwrap(),
                start as u32,
            )
            .unwrap();
    }
    let q = MxArray::random_normal(&[1, 24, query as i64, 256], 0.0, 0.5, Some(DType::BFloat16))
        .unwrap();
    let (gk, gv) = adapter.gather_kv_for_prefill_sdpa(0, context).unwrap();
    let actual = scaled_dot_product_attention_causal(&q, &gk, &gv, 1.0 / 16.0).unwrap();
    let k = k
        .reshape(&[1, context as i64, 4, 256])
        .unwrap()
        .transpose(Some(&[0, 2, 1, 3]))
        .unwrap();
    let v = v
        .reshape(&[1, context as i64, 4, 256])
        .unwrap()
        .transpose(Some(&[0, 2, 1, 3]))
        .unwrap();
    let mask = create_causal_mask(query as i32, Some((context - query) as i32), None).unwrap();
    let reference = scaled_dot_product_attention(&q, &k, &v, 1.0 / 16.0, Some(&mask)).unwrap();
    assert_close(&actual, &reference, "paged write/gather continuation");
    adapter.release_request().unwrap();
}

fn assert_close(actual: &MxArray, expected: &MxArray, label: &str) {
    assert_eq!(
        actual.shape().unwrap().as_ref(),
        expected.shape().unwrap().as_ref()
    );
    let actual = actual.to_float32().unwrap();
    let expected = expected.to_float32().unwrap();
    let mut squared_diff = 0.0f64;
    let mut squared_ref = 0.0f64;
    let mut maximum = 0.0f32;
    for (&a, &b) in actual.iter().zip(expected.iter()) {
        assert!(a.is_finite() && b.is_finite(), "nonfinite output: {label}");
        maximum = maximum.max((a - b).abs());
        squared_diff += f64::from(a - b).powi(2);
        squared_ref += f64::from(b).powi(2);
    }
    let relative_l2 = (squared_diff / squared_ref.max(1e-30)).sqrt();
    eprintln!("{label}: max_abs={maximum:.6}, relative_l2={relative_l2:.6}");
    assert!(
        maximum < 0.02 && relative_l2 < 0.008,
        "{label}: {maximum} / {relative_l2}"
    );
}

fn compare(dtype: DType, batch: i64, qlen: i64, klen: i64, strided: bool) {
    unsafe { mlx_sys::mlx_seed(0xD256_0316) };
    let make = |heads, tokens| {
        let shape = if strided {
            [batch, tokens, heads, 256]
        } else {
            [batch, heads, tokens, 256]
        };
        let tensor = MxArray::random_normal(&shape, 0.0, 0.5, Some(dtype)).unwrap();
        if strided {
            tensor.transpose(Some(&[0, 2, 1, 3])).unwrap()
        } else {
            tensor
        }
    };
    let q = make(24, qlen);
    let k = make(4, klen);
    let v = make(4, klen);
    let actual = scaled_dot_product_attention_causal(&q, &k, &v, 1.0 / 16.0).unwrap();
    let mask = create_causal_mask(qlen as i32, Some((klen - qlen) as i32), None).unwrap();
    let reference = scaled_dot_product_attention(&q, &k, &v, 1.0 / 16.0, Some(&mask)).unwrap();
    let label = format!("{dtype:?} batch={batch} q={qlen} k={klen} strided={strided}");
    assert_close(&actual, &reference, &label);

    // A cached continuation must use the original right-aligned causal
    // diagonal, including when a tail crosses a query tile boundary.
    if qlen >= 33 {
        let split = qlen / 2;
        let q_head = q.slice(&[0, 0, 0, 0], &[batch, 24, split, 256]).unwrap();
        let q_tail = q.slice(&[0, 0, split, 0], &[batch, 24, qlen, 256]).unwrap();
        let end = klen - qlen + split;
        let k_head = k.slice(&[0, 0, 0, 0], &[batch, 4, end, 256]).unwrap();
        let v_head = v.slice(&[0, 0, 0, 0], &[batch, 4, end, 256]).unwrap();
        let a = scaled_dot_product_attention_causal(&q_head, &k_head, &v_head, 1.0 / 16.0).unwrap();
        let b = scaled_dot_product_attention_causal(&q_tail, &k, &v, 1.0 / 16.0).unwrap();
        let joined = MxArray::concatenate(&a, &b, 2).unwrap();
        assert_close(&joined, &reference, &format!("split {label}"));
    }
}

#[test]
#[ignore = "explicit Metal validation; forces the portable path in a fresh process"]
fn portable_d256_matches_reference_and_continuations() {
    if std::env::var_os("MLX_PORTABLE_D256_TEST_CHILD").is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .env("MLX_PORTABLE_D256_TEST_CHILD", "1")
            .env("MLX_PORTABLE_D256_SDPA", "1")
            .env_remove("MLX_ENABLE_D256_FULL_SDPA")
            .args([
                "--ignored",
                "--exact",
                "portable_d256_matches_reference_and_continuations",
                "--nocapture",
            ])
            .status()
            .unwrap();
        assert!(result.success());
        return;
    }
    assert!(
        unsafe { mlx_sys::mlx_metal_portable_d256_sdpa_available() },
        "the portable pipeline must compile; comparing two fallbacks is not validation"
    );
    for dtype in [DType::BFloat16, DType::Float16] {
        for (batch, query, context, strided) in [
            (1, 9, 9, false),
            (1, 31, 63, true),
            (2, 32, 64, false),
            (2, 33, 67, true),
            (1, 127, 4097, true),
            (1, 531, 4131, false),
            (1, 1012, 4112, true),
            (1, 1024, 4096, false),
            (1, 67, 32771, true),
            (1, 531, 32771, false),
        ] {
            compare(dtype, batch, query, context, strided);
            synchronize_and_clear_cache();
        }
    }
    // Unsupported precision must retain the stock path even when forced.
    compare(DType::Float32, 1, 33, 67, true);
    paged_continuation();
}

#[test]
#[ignore = "explicit Metal validation of rollback in a fresh process"]
fn portable_d256_rollback() {
    if std::env::var_os("MLX_PORTABLE_D256_TEST_CHILD").is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .env("MLX_PORTABLE_D256_TEST_CHILD", "1")
            .env("MLX_PORTABLE_D256_SDPA", "0")
            .args([
                "--ignored",
                "--exact",
                "portable_d256_rollback",
                "--nocapture",
            ])
            .status()
            .unwrap();
        assert!(result.success());
        return;
    }
    assert!(!unsafe { mlx_sys::mlx_metal_portable_d256_sdpa_available() });
    compare(DType::BFloat16, 1, 33, 257, true);
}
