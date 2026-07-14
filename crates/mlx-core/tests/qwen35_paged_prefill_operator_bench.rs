#![cfg(target_os = "macos")]

//! Isolated Qwen3.5/Qwen3.6 dense paged-prefill operator benchmark.
//!
//! This test deliberately uses the production [`PagedKVCacheAdapter`] APIs for
//! the graph-native paged K/V write and for both candidate prefill routes. It
//! is ignored because a long-context shape allocates several GiB of unified
//! memory. Run exactly one shape and one route per process:
//!
//! ```text
//! MLX_QWEN35_PREFILL_BENCH_CONTEXT=65536 \
//! MLX_QWEN35_PREFILL_BENCH_QUERY=1024 \
//! MLX_QWEN35_PREFILL_BENCH_ROUTE=sdpa \
//! MLX_QWEN35_PREFILL_BENCH_EXPECT=fused \
//! MLX_QWEN35_PREFILL_BENCH_WARMUP=1 \
//! MLX_QWEN35_PREFILL_BENCH_ITERS=3 \
//! cargo test -p mlx-core --release --test qwen35_paged_prefill_operator_bench \
//!   qwen35_paged_prefill_operator_benchmark -- --ignored --exact --nocapture
//! ```
//!
//! For the direct compact paged-attention arm, set both
//! `MLX_QWEN35_PREFILL_BENCH_ROUTE=varlen` and
//! `MLX_QWEN35_PREFILL_BENCH_EXPECT=varlen`. The varlen arm never falls back
//! to SDPA: an auxiliary buffer that would exceed Metal's `INT_MAX` element
//! limit is reported as a hard benchmark error before the large pool is
//! allocated. An SDPA run must explicitly expect `fused` or `fallback`; the
//! benchmark probes MLX's D=256 eligibility predicate and fails before
//! allocating the pool if the predicted execution does not match.

use std::env;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mlx_core::array::{
    DType, MxArray, clear_cache, get_active_memory, get_peak_memory, reset_peak_memory,
    scaled_dot_product_attention_causal, synchronize, synchronize_and_clear_cache,
};
use mlx_core::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig};

const BLOCK_SIZE: u32 = 16;
const NUM_QUERY_HEADS: u32 = 24;
const NUM_KV_HEADS: u32 = 4;
const HEAD_SIZE: u32 = 256;
const NUM_LAYERS: u32 = 1;
const V2_PARTITION_SIZE: u64 = 512;

const CONTEXT_ENV: &str = "MLX_QWEN35_PREFILL_BENCH_CONTEXT";
const QUERY_ENV: &str = "MLX_QWEN35_PREFILL_BENCH_QUERY";
const ROUTE_ENV: &str = "MLX_QWEN35_PREFILL_BENCH_ROUTE";
const EXPECT_ENV: &str = "MLX_QWEN35_PREFILL_BENCH_EXPECT";
const WARMUP_ENV: &str = "MLX_QWEN35_PREFILL_BENCH_WARMUP";
const ITERS_ENV: &str = "MLX_QWEN35_PREFILL_BENCH_ITERS";

type BenchResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Sdpa,
    Varlen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedExecution {
    Fused,
    Fallback,
    Varlen,
}

impl ExpectedExecution {
    fn from_env() -> BenchResult<Self> {
        let raw = env::var(EXPECT_ENV).map_err(|error| match error {
            env::VarError::NotPresent => format!(
                "{EXPECT_ENV} is required; use 'fused' or 'fallback' for the SDPA route, \
                 or 'varlen' for the varlen route"
            ),
            other => format!("failed to read {EXPECT_ENV}: {other}"),
        })?;
        match raw.to_ascii_lowercase().as_str() {
            "fused" => Ok(Self::Fused),
            "fallback" => Ok(Self::Fallback),
            "varlen" => Ok(Self::Varlen),
            value => Err(format!(
                "{EXPECT_ENV} must be exactly 'fused', 'fallback', or 'varlen'; got {value:?}"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::Fallback => "fallback",
            Self::Varlen => "varlen",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ExecutionProbe {
    predicted_execution: ExpectedExecution,
    d256_full_sdpa_available: bool,
    d256_full_sdpa_would_use: bool,
}

impl Route {
    fn from_env() -> BenchResult<Self> {
        match env::var(ROUTE_ENV)
            .unwrap_or_else(|_| "sdpa".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "sdpa" => Ok(Self::Sdpa),
            "varlen" => Ok(Self::Varlen),
            route => Err(format!(
                "{ROUTE_ENV} must be exactly 'sdpa' or 'varlen'; got {route:?}"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Sdpa => "sdpa",
            Self::Varlen => "varlen",
        }
    }
}

#[derive(Debug)]
struct BenchConfig {
    context: u32,
    query: u32,
    route: Route,
    expected_execution: ExpectedExecution,
    warmup: usize,
    iters: usize,
}

impl BenchConfig {
    fn from_env() -> BenchResult<Self> {
        let config = Self {
            context: parse_env(CONTEXT_ENV, 65_536u32)?,
            query: parse_env(QUERY_ENV, 1_024u32)?,
            route: Route::from_env()?,
            expected_execution: ExpectedExecution::from_env()?,
            warmup: parse_env(WARMUP_ENV, 1usize)?,
            iters: parse_env(ITERS_ENV, 3usize)?,
        };
        if config.context == 0 {
            return Err(format!("{CONTEXT_ENV} must be greater than zero"));
        }
        if config.query == 0 || config.query > config.context {
            return Err(format!(
                "{QUERY_ENV} must be in 1..={}; got {}",
                config.context, config.query
            ));
        }
        if config.iters == 0 {
            return Err(format!("{ITERS_ENV} must be greater than zero"));
        }
        match (config.route, config.expected_execution) {
            (Route::Sdpa, ExpectedExecution::Fused | ExpectedExecution::Fallback) => {}
            (Route::Varlen, ExpectedExecution::Varlen) => {}
            (route, expected) => {
                return Err(format!(
                    "incompatible {ROUTE_ENV}={} and {EXPECT_ENV}={}: SDPA requires fused or \
                     fallback, while varlen requires varlen",
                    route.name(),
                    expected.name()
                ));
            }
        }
        if matches!(config.route, Route::Sdpa) && config.query <= 8 {
            return Err(format!(
                "the SDPA arm benchmarks full attention and requires {QUERY_ENV} > 8; \
                 query lengths <= 8 use MLX's separate vector-attention route"
            ));
        }
        if matches!(config.route, Route::Varlen) {
            reject_oversized_varlen_aux(config.context, config.query)?;
        }
        Ok(config)
    }
}

fn probe_execution(config: &BenchConfig) -> BenchResult<ExecutionProbe> {
    let query_length = i32::try_from(config.query)
        .map_err(|_| format!("{QUERY_ENV}={} exceeds INT32_MAX", config.query))?;
    let key_length = i32::try_from(config.context)
        .map_err(|_| format!("{CONTEXT_ENV}={} exceeds INT32_MAX", config.context))?;

    let mut available = false;
    let available_status =
        unsafe { mlx_sys::mlx_metal_d256_full_sdpa_available(false, &mut available) };
    if available_status != 0 {
        return Err(format!(
            "mlx_metal_d256_full_sdpa_available failed with status {available_status}"
        ));
    }

    let mut would_use = false;
    let predicate_status = unsafe {
        mlx_sys::mlx_metal_d256_full_sdpa_would_use(
            false,
            HEAD_SIZE as i32,
            HEAD_SIZE as i32,
            query_length,
            key_length,
            true,
            false,
            &mut would_use,
        )
    };
    if predicate_status != 0 {
        return Err(format!(
            "mlx_metal_d256_full_sdpa_would_use failed with status {predicate_status}"
        ));
    }

    let predicted_execution = match config.route {
        Route::Sdpa => {
            if would_use {
                ExpectedExecution::Fused
            } else {
                ExpectedExecution::Fallback
            }
        }
        Route::Varlen => ExpectedExecution::Varlen,
    };
    if matches!(config.route, Route::Sdpa) && predicted_execution != config.expected_execution {
        return Err(format!(
            "SDPA execution expectation mismatch before allocation: expected {}, but MLX's \
             D=256 eligibility predicate predicts {} (available={available}, query={}, context={}); \
             check MLX_ENABLE_D256_FULL_SDPA, MLX_ENABLE_TF32, OS, and GPU capability",
            config.expected_execution.name(),
            predicted_execution.name(),
            config.query,
            config.context,
        ));
    }

    Ok(ExecutionProbe {
        predicted_execution,
        d256_full_sdpa_available: available,
        d256_full_sdpa_would_use: would_use,
    })
}

fn parse_env<T>(name: &str, default: T) -> BenchResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(raw) => raw
            .parse::<T>()
            .map_err(|error| format!("invalid {name}={raw:?}: {error}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed to read {name}: {error}")),
    }
}

fn v2_varlen_partition_upper_bound(context: u32, query: u32) -> u64 {
    let generic = u64::from(context).div_ceil(V2_PARTITION_SIZE);
    if query == 2 && context >= 8_192 {
        let grouped = match context {
            0..=4_096 => 32,
            4_097..=8_192 => 64,
            8_193..=16_383 => 128,
            16_384..=32_768 => 256,
            32_769..=65_536 => 512,
            _ => 1_024,
        };
        generic.max(grouped)
    } else {
        generic
    }
}

fn reject_oversized_varlen_aux(context: u32, query: u32) -> BenchResult<()> {
    if context <= V2_PARTITION_SIZE as u32 {
        return Ok(());
    }
    let partitions = v2_varlen_partition_upper_bound(context, query);
    let exp_sums_elements =
        u128::from(query) * u128::from(NUM_QUERY_HEADS) * u128::from(partitions);
    let tmp_out_elements = exp_sums_elements * u128::from(HEAD_SIZE);
    let limit = i32::MAX as u128;
    if exp_sums_elements <= limit && tmp_out_elements <= limit {
        return Ok(());
    }

    let per_query = u128::from(NUM_QUERY_HEADS) * u128::from(partitions) * u128::from(HEAD_SIZE);
    let max_query = limit / per_query;
    Err(format!(
        "direct varlen benchmark rejected: paged-attention V2 auxiliary storage would exceed \
         INT_MAX elements (context={context}, query={query}, partitions={partitions}, \
         exp_sums_elements={exp_sums_elements}, tmp_out_elements={tmp_out_elements}, \
         limit={limit}, maximum_query_for_context={max_query}); no SDPA fallback was run"
    ))
}

fn build_adapter(context: u32) -> BenchResult<PagedKVCacheAdapter> {
    let num_blocks = context.div_ceil(BLOCK_SIZE);
    let pool_bytes = u64::from(num_blocks)
        * u64::from(BLOCK_SIZE)
        * u64::from(NUM_KV_HEADS)
        * u64::from(HEAD_SIZE)
        * 2 // K and V
        * 2; // BF16 bytes
    let gpu_memory_mb = pool_bytes.div_ceil(1024 * 1024).max(256);
    let gpu_memory_mb = u32::try_from(gpu_memory_mb)
        .map_err(|_| format!("pool size {pool_bytes} bytes does not fit gpu_memory_mb"))?;

    let paged_config = PagedAttentionConfig {
        block_size: BLOCK_SIZE,
        gpu_memory_mb,
        head_size: HEAD_SIZE,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        use_fp8_cache: Some(false),
        max_seq_len: Some(context),
        max_batch_size: Some(1),
    };
    let pool = Arc::new(LayerKVPool::new(
        paged_config,
        num_blocks,
        MetalDtype::BFloat16,
    )?);
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(num_blocks, BLOCK_SIZE)));
    let mut adapter = PagedKVCacheAdapter::new(allocator, pool, BLOCK_SIZE)?;
    adapter.reset_for_new_request(1)?;
    adapter.allocate_suffix_blocks(context)?;
    adapter.record_tokens(&vec![0; context as usize])?;
    Ok(adapter)
}

fn graph_native_write(adapter: &mut PagedKVCacheAdapter, context: u32) -> BenchResult<Duration> {
    let shape = [
        i64::from(context),
        i64::from(NUM_KV_HEADS),
        i64::from(HEAD_SIZE),
    ];
    let keys = MxArray::zeros(&shape, Some(DType::BFloat16))
        .map_err(|error| format!("create BF16 keys: {error}"))?;
    let values = MxArray::ones(&shape, Some(DType::BFloat16))
        .map_err(|error| format!("create BF16 values: {error}"))?;

    let started = Instant::now();
    adapter.update_keys_values_native(0, &keys, &values, 0)?;
    adapter.eval_pending_pool_writes()?;
    synchronize();
    Ok(started.elapsed())
}

fn make_queries(query: u32) -> BenchResult<MxArray> {
    MxArray::zeros(
        &[
            i64::from(query),
            i64::from(NUM_QUERY_HEADS),
            i64::from(HEAD_SIZE),
        ],
        Some(DType::BFloat16),
    )
    .map_err(|error| format!("create BF16 queries: {error}"))
}

fn build_route_output(
    adapter: &mut PagedKVCacheAdapter,
    queries: &MxArray,
    config: &BenchConfig,
) -> BenchResult<MxArray> {
    let scale = 1.0 / f64::from(HEAD_SIZE).sqrt();
    match config.route {
        Route::Sdpa => {
            let (keys, values) = adapter.gather_kv_for_prefill_sdpa(0, config.context)?;
            let queries = queries
                .reshape(&[
                    1,
                    i64::from(config.query),
                    i64::from(NUM_QUERY_HEADS),
                    i64::from(HEAD_SIZE),
                ])
                .map_err(|error| format!("reshape SDPA queries: {error}"))?
                .transpose(Some(&[0, 2, 1, 3]))
                .map_err(|error| format!("transpose SDPA queries: {error}"))?;
            scaled_dot_product_attention_causal(&queries, &keys, &values, scale)
                .map_err(|error| format!("causal SDPA: {error}"))
        }
        Route::Varlen => adapter.gather_kv_for_prefill_chunk_varlen(
            0,
            queries,
            config.context - config.query,
            scale as f32,
        ),
    }
}

fn evaluate_once(
    adapter: &mut PagedKVCacheAdapter,
    queries: &MxArray,
    config: &BenchConfig,
) -> BenchResult<(Duration, MxArray)> {
    let started = Instant::now();
    let output = build_route_output(adapter, queries, config)?;
    let mut output_handle = output.as_raw_ptr();
    if !unsafe { mlx_sys::mlx_eval(&mut output_handle, 1) } {
        return Err(format!(
            "{} evaluation failed; no fallback result was benchmarked",
            config.route.name()
        ));
    }
    synchronize();
    Ok((started.elapsed(), output))
}

fn verify_all_ones(output: &MxArray, route: Route) -> BenchResult<()> {
    let values = output
        .to_float32()
        .map_err(|error| format!("copy {} output for verification: {error}", route.name()))?;
    let mut maximum_error = 0.0f32;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{} produced non-finite output at element {index}: {value}",
                route.name()
            ));
        }
        maximum_error = maximum_error.max((value - 1.0).abs());
    }
    if maximum_error > 0.02 {
        return Err(format!(
            "{} deterministic Q=K=0,V=1 check failed: maximum absolute error \
             {maximum_error:.6} exceeds 0.02",
            route.name()
        ));
    }
    Ok(())
}

fn run_benchmark() -> BenchResult<()> {
    let config = BenchConfig::from_env()?;
    // This probe is intentionally before `build_adapter`: a stale A/B
    // expectation must not allocate a multi-GiB paged pool before failing.
    let execution_probe = probe_execution(&config)?;
    let mut adapter = build_adapter(config.context)?;
    let write_elapsed = graph_native_write(&mut adapter, config.context)?;
    synchronize_and_clear_cache();

    let queries = make_queries(config.query)?;
    for _ in 0..config.warmup {
        let (_, output) = evaluate_once(&mut adapter, &queries, &config)?;
        drop(output);
        synchronize_and_clear_cache();
    }

    reset_peak_memory();
    let active_memory_before = get_active_memory();
    let mut samples = Vec::with_capacity(config.iters);
    let mut final_output = None;
    for iteration in 0..config.iters {
        let (elapsed, output) = evaluate_once(&mut adapter, &queries, &config)?;
        samples.push(elapsed);
        if iteration + 1 == config.iters {
            final_output = Some(output);
        } else {
            drop(output);
            synchronize_and_clear_cache();
        }
    }
    let active_memory_after = get_active_memory();
    let peak_memory = get_peak_memory();

    verify_all_ones(
        final_output
            .as_ref()
            .ok_or_else(|| "benchmark did not retain a final output".to_string())?,
        config.route,
    )?;

    samples.sort_unstable();
    let median = if samples.len().is_multiple_of(2) {
        (samples[samples.len() / 2 - 1] + samples[samples.len() / 2]) / 2
    } else {
        samples[samples.len() / 2]
    };
    let median_ms = median.as_secs_f64() * 1_000.0;
    let tokens_per_second = f64::from(config.query) / median.as_secs_f64();
    let sample_ms: Vec<f64> = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1_000.0)
        .collect();

    println!(
        "QWEN35_PAGED_PREFILL_BENCH {}",
        serde_json::json!({
            "route": config.route.name(),
            "expected_execution": config.expected_execution.name(),
            "predicted_execution": execution_probe.predicted_execution.name(),
            "d256_full_sdpa_available": execution_probe.d256_full_sdpa_available,
            "d256_full_sdpa_would_use": execution_probe.d256_full_sdpa_would_use,
            "dtype": "bf16",
            "batch": 1,
            "query_heads": NUM_QUERY_HEADS,
            "kv_heads": NUM_KV_HEADS,
            "head_size": HEAD_SIZE,
            "block_size": BLOCK_SIZE,
            "context_tokens": config.context,
            "query_tokens": config.query,
            "warmup_iterations": config.warmup,
            "timed_iterations": config.iters,
            "write_ms": write_elapsed.as_secs_f64() * 1_000.0,
            "sample_ms": sample_ms,
            "median_synchronized_wall_ms": median_ms,
            "tokens_per_second": tokens_per_second,
            "active_memory_before_bytes": active_memory_before,
            "active_memory_after_bytes": active_memory_after,
            "peak_memory_bytes": peak_memory,
            "verified": "Q=K=0,V=1",
        })
    );

    drop(final_output);
    clear_cache();
    synchronize();
    adapter.release_request()?;
    Ok(())
}

#[test]
#[ignore = "manual release-only Metal benchmark; allocates one long-context shape per process"]
fn qwen35_paged_prefill_operator_benchmark() {
    if !cfg!(not(debug_assertions)) {
        panic!("this benchmark must be compiled with --release");
    }
    run_benchmark().unwrap_or_else(|error| panic!("{error}"));
}
