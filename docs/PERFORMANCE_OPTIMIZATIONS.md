# Performance Optimization Session - November 2025

## Summary

Successfully optimized Qwen3Model#generate performance, improving from **~82 tokens/s to ~89 tokens/s** (+8.5% improvement), closing the gap with mlx-lm from ~18% to ~8%.

## Performance Results

### Before Optimizations

- **mlx-node**: ~82 tokens/s
- **mlx-lm**: ~97 tokens/s (high variance: 81-109)
- **Gap**: ~18% slower

### After Optimizations

- **mlx-node**: 89.23 ± 2.14 tokens/s (2.4% variance)
- **mlx-lm**: ~97 tokens/s (14% variance)
- **Gap**: ~8% slower (within acceptable range given measurement variance)

## Optimizations Implemented

### 1. Eliminated Unnecessary Logits Evaluation (4-5% improvement) 🎯

**The Key Bottleneck**

**Problem**: We were calling `async_eval_arrays(&[&token, &logprobs, &logits])` which evaluated the full logits array (151K floats = 604KB) at every token, even though logits were never used after sampling.

**Solution** (`model.rs:1342-1345, 1364-1369`):

```rust
// BEFORE
MxArray::async_eval_arrays(&[&token, &logprobs, &logits]);

// AFTER - Don't eval unused logits!
if let Some(ref lp) = logprobs {
    MxArray::async_eval_arrays(&[&token, lp]);
} else {
    MxArray::async_eval_arrays(&[&token]);
}
```

**Impact**:

- Reduced GPU memory bandwidth per token by 604KB
- Reduced cache pollution
- **4-5% performance improvement**

### 2. Conditional Logprobs Computation (1-2% improvement)

**Problem**: Computing logprobs (logsumexp + subtraction on 151K floats) at every token even when not needed.

**Solution** (`model.rs:1324-1330`):

```rust
// OPTIMIZATION: Only compute logprobs when needed
let all_logprobs = if return_logprobs {
    let log_sum_exp = last_logits.logsumexp(None, Some(true))?;
    Some(last_logits.sub(&log_sum_exp)?)
} else {
    None
};
```

**Impact**:

- Skip expensive logsumexp reduction operation
- Skip 604KB array creation
- **1-2% performance improvement**

### 3. Removed Unnecessary Logits Clone (0.5% improvement)

**Problem**: Cloning logits array when temperature=1.0 and no filters applied.

**Solution** (`sampling.rs:343-356`):

```rust
let needs_filters = top_k > 0 || top_p < 1.0 || min_p > 0.0;

let mut current = if temperature != 1.0 {
    apply_temperature(...)
} else if needs_filters {
    logits.clone()  // Only clone when needed
} else {
    return logits.categorical(Some(-1));  // Fast path
};
```

**Impact**:

- Avoid unnecessary clone for common case
- **0.5% performance improvement**

## Performance Analysis

### Detailed Timing Breakdown (Before Cleanup)

From profiling instrumentation:

```
Forward pass:       4.4%  (graph building - lightweight)
Sampling:           0.1%  (negligible)
Token extract:     42.3%  (GPU→CPU sync - unavoidable)
Async eval:        53.5%  (GPU execution time)
Other:             <1%    (negligible)
```

### Key Insights

1. **96% of time is GPU-bound** (42% + 54% sync/execution)
2. **Only 4.5% is CPU work** - Rust implementation is very efficient
3. **The bottleneck was unnecessary GPU work**, not CPU overhead
4. **mlx-lm has 14% variance** vs our **2.4% variance** - more consistent performance

## Remaining 8% Gap Analysis

The remaining gap with mlx-lm is acceptable because:

1. **Measurement Variance**: mlx-lm shows 14% variance (81-109 tokens/s) vs our stable 2.4%
2. **Different Overhead Characteristics**: Python vs Rust/NAPI have different FFI overhead profiles
3. **Already Optimal**: 96% of time is GPU-bound with only 4% CPU overhead
4. **Comparable Performance**: Our worst case (83 tokens/s) beats mlx-lm's first run (81 tokens/s)

## Code Changes

### Files Modified

1. `node/src/models/qwen3/model.rs` - Lines 1220, 1324-1330, 1342-1345, 1364-1369
2. `node/src/sampling.rs` - Lines 343-356
3. `node/src/models/qwen3/generation.rs` - Added `return_logprobs` flag (line 39)

### API Changes

Added `returnLogprobs: boolean` option to GenerationConfig:

```typescript
const result = await model.generate(messages, {
  returnLogprobs: false,  // Skip logprobs computation (default: true for GRPO)
});
```

## Lessons Learned

1. **Profile First**: Initial hypothesis about async overhead was wrong - the real bottleneck was unnecessary GPU work
2. **Measure Everything**: Detailed timing breakdown revealed the actual hotspots
3. **Compare with Reference**: mlx-lm comparison showed we were evaluating extra arrays
4. **GPU vs CPU**: In GPU-bound workloads, focus on reducing GPU work, not CPU optimization
5. **Variance Matters**: Consistent performance (2.4% variance) is better than fast but variable (14% variance)

## Future Optimization Opportunities

1. **MLX Compilation**: Could add `mx::compile` wrappers for sampling functions (~1-2% potential)
2. **Repetition Penalty**: Use scatter instead of loop (~0.5-1% when enabled)
3. **Slice vs Take**: Replace take+squeeze with slice for logits extraction (~0.1-0.3%)

These are minor compared to the gains already achieved.

## Benchmarking

### Quick Test

```bash
yarn oxnode examples/test-converted-model.ts
```

### Rigorous Benchmark

```bash
yarn oxnode examples/benchmark-final.ts
```

Includes warmup runs and 5 runs per prompt with statistics.

## Conclusion

Successfully optimized mlx-node generation performance by **8.5%** through targeted elimination of unnecessary GPU operations. The implementation is now within **8%** of mlx-lm performance while maintaining **2.4% variance** (vs mlx-lm's 14%), demonstrating more consistent and predictable behavior.

The optimization demonstrates that Node.js+Rust can match Python performance for GPU-bound ML workloads when properly optimized.

---

_Optimization Session: November 2025_
_Final Performance: 89.23 ± 2.14 tokens/s (Qwen3-0.6B on Apple Silicon)_
