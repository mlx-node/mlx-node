//! Attention VJP Performance Benchmarks
//!
//! Benchmarks for scaled dot product attention backward pass (VJP) performance.
//! These benchmarks measure the time to compute gradients for Q, K, V tensors
//! through the attention mechanism.
//!
//! Run with: cargo bench --package mlx-core --bench attention_vjp_bench

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mlx_core::array::{DType, MxArray, scaled_dot_product_attention};
use mlx_core::autograd::value_and_grad;

/// Helper to create a random tensor with given shape and dtype
fn random_tensor(shape: &[i64], dtype: DType) -> MxArray {
    let arr = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
    if dtype != DType::Float32 {
        arr.astype(dtype).unwrap()
    } else {
        arr
    }
}

/// Benchmark attention VJP computation for given dimensions
fn bench_attention_vjp(batch: i64, heads: i64, seq_len: i64, head_dim: i64, dtype: DType) {
    let scale = 1.0 / (head_dim as f64).sqrt();

    let q = random_tensor(&[batch, heads, seq_len, head_dim], dtype);
    let k = random_tensor(&[batch, heads, seq_len, head_dim], dtype);
    let v = random_tensor(&[batch, heads, seq_len, head_dim], dtype);
    q.eval();
    k.eval();
    v.eval();

    let q_clone = q.copy().unwrap();
    let k_clone = k.copy().unwrap();
    let v_clone = v.copy().unwrap();

    let (_, grads) = value_and_grad(vec![&q_clone, &k_clone, &v_clone], move |params| {
        let attn_out =
            scaled_dot_product_attention(&params[0], &params[1], &params[2], scale, None)?;
        let loss = attn_out.sum(None, None)?;
        Ok(loss)
    })
    .expect("value_and_grad failed");

    // Force evaluation
    grads[0].eval();
    grads[1].eval();
    grads[2].eval();

    // Use black_box to prevent optimization
    black_box(&grads);
}

/// Performance benchmarks for attention VJP at different scales
fn attention_vjp_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_vjp");

    // Small dimensions: batch=1, heads=4, seq=32, head_dim=64
    group.bench_function(BenchmarkId::new("f32", "small"), |b| {
        b.iter(|| bench_attention_vjp(1, 4, 32, 64, DType::Float32))
    });

    // Medium dimensions: batch=2, heads=8, seq=128, head_dim=64
    group.bench_function(BenchmarkId::new("f32", "medium"), |b| {
        b.iter(|| bench_attention_vjp(2, 8, 128, 64, DType::Float32))
    });

    // Large dimensions: batch=4, heads=16, seq=256, head_dim=128
    group.bench_function(BenchmarkId::new("f32", "large"), |b| {
        b.iter(|| bench_attention_vjp(4, 16, 256, 128, DType::Float32))
    });

    group.finish();
}

/// BFloat16 vs Float32 comparison benchmarks
fn bf16_vs_f32_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_vjp_dtype_comparison");

    let batch = 2;
    let heads = 8;
    let seq_len = 64;
    let head_dim = 64;

    group.bench_function(BenchmarkId::new("dtype", "f32"), |b| {
        b.iter(|| bench_attention_vjp(batch, heads, seq_len, head_dim, DType::Float32))
    });

    group.bench_function(BenchmarkId::new("dtype", "bf16"), |b| {
        b.iter(|| bench_attention_vjp(batch, heads, seq_len, head_dim, DType::BFloat16))
    });

    group.finish();
}

criterion_group!(benches, attention_vjp_benchmarks, bf16_vs_f32_benchmarks);
criterion_main!(benches);
