# Portable Qwen prefill on older Apple GPUs

Investigated on 2026-09-22 for an M3 Pro with 36 GB running
`Qwen3.8-27B-UD-Q4_K_XL`. A matched M3 throughput improvement remains unverified.
Local kernel and model validation used an M5 Max with 128 GB.

## Findings and changes

- **Long-context attention:** the M3 trace took 81.058 seconds to prefill 1,026
  new tokens in a 79,038-token conversation. Prefix reuse worked, but D256
  attention fell back to memory-heavy score matrices or repeated paged reads.
  The new causal FP16/BF16 SIMD kernel uses a 32×16×256 tile with 29,184 bytes
  of threadgroup scratch. It checks actual pipeline limits and gives the paged
  planner a fused-memory estimate for supported layouts and more than eight
  query tokens. Its private preamble includes the wide-head synchronization
  fix from [MLX #4185](https://github.com/ml-explore/mlx/pull/4185).
- **Prefill projections:** eligible Q4_K/Q5_K/Q6_K/IQ4_XS matrices use 16-bit
  operands with FP32 accumulation and a 64×64×32 tile (10 KiB scratch).
  Selection requires contiguous FP16/BF16 operands, M≥128, N≥1024, K divisible
  by 256, and at least 512 stock 32×32 tiles. Packed weights retain their
  representation; no persistent dense-weight copy is added.
- **Zero-byte cache budget:** dense and MoE loaders retained replaced source
  tensors while probing memory. They now release those temporary owners and
  clear the allocator cache before sizing the paged pool. Weight accounting,
  memory limits and safety reserves are preserved; failures include the memory
  budget breakdown.

Both kernels select automatically on supported Metal devices without NAX;
unsupported inputs or pipelines retain their existing fallback. The automatic
NAX path is unchanged. `MLX_PORTABLE_D256_SDPA=0` and `MLX_PORTABLE_KQUANT=0`
independently disable the new paths; `=1` forces eligible paths for validation.
Chunk sizes and memory reserves were not raised to compensate for slow prefill.

## Measurements and limitations

These measurements are from the **M5 Max**, forcing portable attention. They
establish operator behavior, not M3 or whole-model throughput. At 79,024 context
and 1,012 query tokens, two fresh processes per route, each with one warmup and
three completed samples, measured:

| Gather + attention route | Median per process | MLX allocator peak |
| ------------------------ | ------------------ | ------------------ |
| Existing paged varlen    | 1,122–1,211 ms     | 1,891.3 MiB        |
| Portable SIMD attention  | 221–231 ms         | 641.7 MiB          |

The peak excludes the external paged pool, whose geometry was identical.
Portable attention was not always fastest: at 32,768 context / 67 queries it
measured 18.9 ms, versus 13.3 ms for unfused SDPA and 28.9 ms for paged varlen.

Releasing temporary weights reduced the load-time active-memory probe from
22,152 to 17,106 MiB on the same 27B checkpoint. A restricted local budget
reproduced the zero-block failure before the fix and passed afterward; tighter
budgets still failed safely.

QMM timing controls were noisy on the shared machine, and one M=128 Q5_K case
was slower. A diagnostic pre-NAX profile attributed 92% of measured layer-stage
time to quantized projections, but its added evaluation barriers alter scheduling.
Staged GDN and alternative matrix tiles did not establish further gains and
were not adopted. Selecting pre-NAX kernels on M5 does not emulate M3 hardware.

### Community context

The reported M3 result was 89.1 prefill / 10.1 decode tokens/s and 16.49 seconds
TTFT, with about 1.5k new + 4.6k cached tokens. The following oMLX submissions
use M3 Pro 18-core GPUs, 36 GB and **cold 4,096-token prompts**:

| Qwen3.8-27B quant | MTP           | Prefill tok/s | Decode tok/s | Source                                                           |
| ----------------- | ------------- | ------------- | ------------ | ---------------------------------------------------------------- |
| MLX 4bit          | Off           | 99.6          | 7.7          | [oMLX 0.6.2](https://omlx.ai/benchmarks/performance/oszqozon)    |
| oQ4e FP16         | Off           | 108.2         | 8.2          | [oMLX 0.6.3rc2](https://omlx.ai/benchmarks/performance/yeoz4lwo) |
| oQ4e FP16         | Lightning MTP | 106.7         | 16.2         | [oMLX 0.6.0](https://omlx.ai/benchmarks/performance/d4wsz7gd)    |

These are different quants and workloads, not controlled runtime or MTP A/B
comparisons. No exact UD-Q4_K_XL match was found. The earlier 77 tokens/s report
and later 89.1 tokens/s report also do not establish a test-build speedup.

## Validation and reproduction

Recorded local validation passed:

- Native build and packaged-metallib checks; Clippy with warnings denied,
  Rust formatting and benchmark lint checks.
- 25 attention/planner tests and 39 D256 numerical comparisons across FP16/BF16,
  ragged and cached shapes, batches, strides, paged writes/gathers and rollback.
- Nine K-quant guard tests, including 24 new CPU-reference cases, plus 32 exact
  stock-SIMD/candidate matrix comparisons; 23 budget tests and 206 persistence
  tests, with three checkpoint-dependent cases ignored.
- Full-checkpoint AR and MTP cold-prompt/continuation checks preserved outputs
  and cache reuse within each mode. This does not establish AR-versus-MTP parity.

Run the retained regression tests from the repository root:

```sh
vp run build:native
cargo test -p mlx-core --release --test portable_d256_sdpa -- --ignored --nocapture --test-threads=1
MLX_METAL_GPU_ARCH=applegpu_g15s MLX_PORTABLE_KQUANT=1 \
  cargo test -p mlx-core --release --test kquant_mode_guards -- --nocapture
```

The [paged-prefill operator benchmark](../../../crates/mlx-core/tests/qwen35_paged_prefill_operator_bench.rs)
documents shape, route, warmup and iteration controls for repeating operator measurements.

For a real M3 A/B, use fresh processes and the same checkpoint, cold prompt,
cached follow-up, output length and MTP setting. Compare each portable path
against its rollback separately; leave architecture and memory-limit overrides
unset. Record app/native revisions, GPU cores, thermals, swap, cache hits,
output hashes, TTFT, prefill and decode rates. Repeat runs before claiming a
throughput improvement.
