#![cfg(target_os = "macos")]

//! Numerical coverage for the NAX fused causal SDPA path used by
//! Qwen3.5/Qwen3.6 dense full-attention layers (`D=256`).

use mlx_core::array::mask::create_causal_mask;
use mlx_core::array::{
    DType, MxArray, scaled_dot_product_attention_causal, synchronize_and_clear_cache,
};
use mlx_core::nn::Activations;

const TF32_CHILD_ENV: &str = "MLX_D256_TF32_TEST_CHILD";

const BATCH: i64 = 1;
const QUERY_HEADS: i64 = 24;
const KV_HEADS: i64 = 4;
const HEAD_SIZE: i64 = 256;

fn d256_available(effective_dtype_is_float32: bool) -> bool {
    let mut available = false;
    let status = unsafe {
        mlx_sys::mlx_metal_d256_full_sdpa_available(effective_dtype_is_float32, &mut available)
    };
    assert_eq!(status, 0, "D=256 capability probe failed");
    available
}

fn assert_fused_matches_explicit_mask(query_len: i64, key_len: i64) {
    assert!(d256_available(false), "NAX BF16 D=256 route is required");
    assert!(query_len >= 1_024);
    assert!(key_len >= query_len);
    assert!(would_use_fused_d256(query_len, key_len));

    unsafe { mlx_sys::mlx_seed(0xD256_5DFA) };
    let queries = MxArray::random_uniform(
        &[BATCH, QUERY_HEADS, query_len, HEAD_SIZE],
        -0.25,
        0.25,
        Some(DType::BFloat16),
    )
    .expect("queries");
    let keys = MxArray::random_uniform(
        &[BATCH, KV_HEADS, key_len, HEAD_SIZE],
        -0.25,
        0.25,
        Some(DType::BFloat16),
    )
    .expect("keys");
    let values = MxArray::random_uniform(
        &[BATCH, KV_HEADS, key_len, HEAD_SIZE],
        -0.0625,
        0.0625,
        Some(DType::BFloat16),
    )
    .expect("values");
    let scale = 1.0 / (HEAD_SIZE as f64).sqrt();

    // No array mask: this takes the fused D=256 path on NAX for q >= 1024.
    let fused = scaled_dot_product_attention_causal(&queries, &keys, &values, scale)
        .expect("fused causal SDPA");

    let reference = primitive_reference(&queries, &keys, &values, scale, query_len, key_len);

    let expected_shape = [BATCH, QUERY_HEADS, query_len, HEAD_SIZE];
    let fused_shape = fused.shape().expect("fused shape");
    let reference_shape = reference.shape().expect("reference shape");
    assert_eq!(fused_shape.as_ref(), expected_shape);
    assert_eq!(reference_shape.as_ref(), expected_shape);

    let fused = fused.to_float32().expect("fused values");
    let reference = reference.to_float32().expect("reference values");
    assert_eq!(fused.len(), reference.len());

    let mut max_abs = 0.0_f64;
    let mut diff_sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    for (index, (&actual, &expected)) in fused.iter().zip(reference.iter()).enumerate() {
        assert!(
            actual.is_finite(),
            "fused output[{index}] is non-finite: {actual}"
        );
        let diff = f64::from(actual - expected);
        max_abs = max_abs.max(diff.abs());
        diff_sq += diff * diff;
        ref_sq += f64::from(expected) * f64::from(expected);
    }
    let relative_l2 = (diff_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        max_abs <= 0.02,
        "D=256 fused SDPA max_abs={max_abs:.6} exceeds 0.02 \
         (q={query_len}, k={key_len}, relative_l2={relative_l2:.6})"
    );
    assert!(
        relative_l2 <= 0.005,
        "D=256 fused SDPA relative_l2={relative_l2:.6} exceeds 0.005 \
         (q={query_len}, k={key_len}, max_abs={max_abs:.6})"
    );
    synchronize_and_clear_cache();
}

/// FP32 attention from primitives (MLX's unfused graph). Any `fast::` SDPA
/// call, even with an explicit mask, now takes the fused NAX kernel at q >= 1024.
fn primitive_reference(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    scale: f64,
    query_len: i64,
    key_len: i64,
) -> MxArray {
    let repeats = i32::try_from(QUERY_HEADS / KV_HEADS).expect("GQA factor");
    let f32_heads = |x: &MxArray| {
        x.astype(DType::Float32)
            .and_then(|x| x.repeat(repeats, 1))
            .expect("GQA-expanded FP32 operand")
    };
    let offset = i32::try_from(key_len - query_len).expect("causal offset");
    let keep = create_causal_mask(
        i32::try_from(query_len).expect("query length"),
        Some(offset),
        None,
    )
    .expect("causal mask");
    let additive = keep
        .astype(DType::Float32)
        .and_then(|m| m.sub_scalar(1.0))
        .and_then(|m| m.mul_scalar(1e9))
        .expect("additive mask");
    let scores = queries
        .astype(DType::Float32)
        .and_then(|q| q.mul_scalar(scale))
        .and_then(|q| q.matmul(&f32_heads(keys).transpose(Some(&[0, 1, 3, 2]))?))
        .and_then(|s| s.add(&additive))
        .expect("scores");
    Activations::softmax(&scores, Some(-1))
        .and_then(|p| p.matmul(&f32_heads(values)))
        .expect("primitive attention reference")
}

fn would_use_fused_d256(query_len: i64, key_len: i64) -> bool {
    would_use_fused_d256_for_dtype(false, query_len, key_len)
}

fn would_use_fused_d256_for_dtype(
    effective_dtype_is_float32: bool,
    query_len: i64,
    key_len: i64,
) -> bool {
    would_use_fused_d256_with(effective_dtype_is_float32, query_len, key_len, true, false)
}

fn would_use_fused_d256_with(
    effective_dtype_is_float32: bool,
    query_len: i64,
    key_len: i64,
    do_causal: bool,
    has_array_mask: bool,
) -> bool {
    let mut would_use = false;
    let status = unsafe {
        mlx_sys::mlx_metal_d256_full_sdpa_would_use(
            effective_dtype_is_float32,
            HEAD_SIZE as i32,
            HEAD_SIZE as i32,
            i32::try_from(query_len).expect("query length"),
            i32::try_from(key_len).expect("key length"),
            do_causal,
            has_array_mask,
            &mut would_use,
        )
    };
    assert_eq!(status, 0, "D=256 route probe failed");
    would_use
}

#[test]
#[ignore = "requires a NAX Metal GPU; explicit execution must fail when the fused route is unavailable"]
fn d256_fused_sdpa_matches_fallback_across_boundaries() {
    assert!(!would_use_fused_d256(1_023, 8_192));
    assert!(would_use_fused_d256_with(false, 1_024, 8_192, false, true));
    assert!(!would_use_fused_d256_with(
        false, 2_048, 8_192, false, false
    ));

    for (query_len, key_len) in [
        (1_024, 8_192),
        // Ragged Q and K tiles with a non-zero causal diagonal offset.
        (1_031, 4_129),
        (2_049, 4_129),
    ] {
        assert_fused_matches_explicit_mask(query_len, key_len);
    }
}

#[test]
#[ignore = "requires a NAX Metal GPU with TF32 enabled for the parent baseline"]
fn d256_float32_route_honors_tf32_gate_in_fresh_process() {
    if std::env::var_os(TF32_CHILD_ENV).is_some() {
        let mut available = true;
        let status = unsafe { mlx_sys::mlx_metal_d256_full_sdpa_available(true, &mut available) };
        assert_eq!(status, 0, "TF32 child capability probe failed");
        assert!(!available, "FP32 D=256 capability requires TF32");
        assert!(!would_use_fused_d256_for_dtype(true, 2_048, 8_192));
        return;
    }
    assert!(
        d256_available(true),
        "TF32 rollback proof requires an available FP32 D=256 baseline"
    );

    let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .env("MLX_ENABLE_TF32", "0")
        .env(TF32_CHILD_ENV, "1")
        .arg("--ignored")
        .arg("--exact")
        .arg("d256_float32_route_honors_tf32_gate_in_fresh_process")
        .arg("--nocapture")
        .output()
        .expect("run TF32 child process");
    assert!(
        output.status.success(),
        "TF32 child failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
