# Gemma 4 GGUF inference optimization

The new Gemma 4 GGUF Q4 decode kernel reads packed weights and original FP16 scales directly, eliminating repeated metadata conversions. Sliding attention now limits reads to the active window. Attention partitions and CPU submission depth are selected from completed decode timings on the current device. There is no new M5 identifier, exact model-size condition, or replay-length threshold in the automatic tuning policy.

The measured model is `gemma-4-12b-it-qat-q4_0.gguf`. It contains 328 Q4_0 projections and a Q6_K tied embedding/output head. Existing GGUF loading and inference are the supported scope; this change does not add a K-quant producer. [The original benchmark](../gemma4-agent-benchmark-2026-09-10/README.md) and [the detailed code-path audit](../gemma4-path-audit-2026-09-10/README.md) establish the starting point.

## Recorded agent benchmark

Every cell below is the median of three fresh-process runs. Both runtimes consume the same token IDs from complete historical coding-agent turn boundaries, use BF16 KV, greedy autoregressive decoding, high thinking, 512-token physical prefill chunks, and produce 256 tokens with zero prompt-cache hits. Each process first warms up with 32 tokens from the real 7,733-token fixture, then clears its prompt cache. Runs alternate runtimes, with a 20-second cooldown. Any calibration during the measured request is included in request and decode timing; both runtimes exclude the same fixed 32-token warmup. No extra benchmark tokens, fabricated tool results, padding, or historical tool execution are used.

| Input tokens | Prompt tok/s MLX / llama.cpp | Decode tok/s MLX / llama.cpp | Request seconds MLX / llama.cpp | Decode speedup over original MLX |
| ---: | ---: | ---: | ---: | ---: |
| 7,733 | 1,153.2 / 1,252.0 | 50.53 / 48.70 | 11.76 / 11.24 | 1.79× |
| 40,528 | 1,000.4 / 740.9 | 39.84 / 35.29 | 46.99 / 61.93 | 2.47× |
| 66,904 | 872.1 / 526.5 | 34.05 / 28.80 | 84.40 / 135.93 | 2.79× |

The machine is an M5 Max with 40 GPU cores and 128 GB of unified memory, connected to AC power. It remained an active desktop; no other inference benchmark or compilation ran during measured samples. macOS reported no thermal or performance warning. Three repeats establish an observed range, not statistical confidence or performance on other hardware. The original MLX comparison was measured earlier on the same machine and fixture; the adjacent llama.cpp columns are newly measured controls.

The checkpoint SHA-256 is `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`. llama.cpp is pinned to `a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92`, and MLX to `6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb`. The tested native addon SHA-256 is `2f380276a3fa2e3dc39f3e5ff377bc213152554d34d2ba3d819db733d3adface`. [All samples and provenance](./results.json) contain min/max values, timing accounting, memory snapshots, generated-output hashes, and the selected plans. The source patch includes both tracked modifications and new source files; its hash is recorded alongside individual file hashes. Before publication, rustfmt reordered the `decode_tuning` and `decoder_layer` declarations in `gemma4/mod.rs`; the recorded hashes retain the measured source, and no runtime logic changed after measurement.

## What changed

**[Direct mixed-dtype Q4 decode](../../../crates/mlx-sys/src/mlx_affine_qmv.cpp).** MLX's general QMM promotes BF16 activations combined with FP16 scale/bias metadata to FP32. Previously it rebuilt FP32 scale and bias arrays for every decode call. Hoisting them at load time removed those repeated conversions but still streamed two FP32 sidecars. The new narrow Metal path reads packed Q4 values and original FP16 scales, widens values in registers, accumulates in FP32, and stores BF16 output. For metadata-confirmed symmetric Q4_0 groups, it derives the bias as `-8 × scale` in registers with the original FP16 rounding. Asymmetric quantization overrides retain their explicit biases.

The kernel is restricted to affine Q4/group32 single-row BF16 inference with compatible matrix shapes. Other bit widths, K-quant packs, data types, shapes, prefill, and gradient transforms retain their existing implementations. The Q6_K head is not misinterpreted as affine Q4. The prepared GGUF cache now records the nested text dtype correctly and invalidates the older Gemma cache layout, so the prefill metadata hoist actually activates. Weights are never reconstructed as a full BF16 matrix by the new decode path.

**[Bounded sliding attention](../../../crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs).** Sliding decode removes retired whole pages from its private read view and rebases only the kernel's local sequence length. Prefill gathers only the necessary window plus the current chunk; the mask uses the corresponding private origin. Absolute RoPE positions, cache ownership, logical token history, and prefix reuse remain unchanged. For this checkpoint's 1,024-token window and 512-token chunks, the gathered width is at most 1,536 tokens rather than the entire 66K history. A boundary key is retained and masked to preserve tile alignment.

**[Device-local scheduling and attention selection](../../../crates/mlx-core/src/models/gemma4/decode_tuning.rs).** The model measures the interval from forward submission until the existing sampled-token completion hook. This includes useful GPU work, CPU graph construction, and launch overhead; it is deliberately not presented as a GPU timestamp. It does not add synchronization or evaluate another forward. The legal stripe candidates are derived from the grouped kernel's supported power-of-two range and available 16-token work tiles. Submission candidates are powers of two bounded by the actual layer count, including no early submission and the final usable prefix.

The search first chooses attention partitions, then early-submission depth, then rechecks neighboring partitions under the chosen schedule. Each candidate receives one unscored first-use step and three measured steps, with alternate rounds reversed. A change must improve the median by more than both measured jitter and a 1% stability floor. This is a finite coordinate search over supported implementations, not a proof of a global optimum. It can choose different plans under different device speeds or background load. Decisions are cached for at most eight power-of-two context scales within one loaded model, are not persisted to disk, and do not transfer a profile between machines. A new context scale calibrates independently; the policy does not continuously track thermal changes after a scale is calibrated.

The learned path currently applies to ordinary single-row Gemma paged AR decoding. Batched scheduling, speculative verification, and other families retain their existing policies. Metal capability, dtype, head geometry, cache layout, and scratch bounds still gate every grouped dispatch. Explicit diagnostic overrides take precedence. The stripe count is captured in the lazy primitive's state and compile identity and reaches the actual graph-native kernel; it is not merely a value in a log. Raw fallback uses its existing conservative policy.

The following are observations from these runs, **not shipped hardware presets**:

| Fixture boundary | Repeat | Context scale | Grouped stripes | Early submitted layers |
| --- | ---: | ---: | ---: | ---: |
| diff-review | 1 | 8,192 | 64 | 32 |
| diff-review | 2 | 8,192 | 64 | 8 |
| diff-review | 3 | 8,192 | 64 | 0 |
| test-review | 1 | 65,536 | 256 | 4 |
| test-review | 2 | 65,536 | 256 | 8 |
| test-review | 3 | 65,536 | 256 | 16 |
| final-review | 1 | 131,072 | 128 | 2 |
| final-review | 2 | 131,072 | 256 | 4 |
| final-review | 3 | 131,072 | 128 | 8 |

The MLX submodule is unchanged. [Exploratory measurements](./experiments.json) retain the pilot and screening results separately from the final matched matrix; fixed submission sweeps and alternative kernels are not shipped device presets.

## Bandwidth evidence and remaining limits

The projection probe replays all 328 real captured single-token activations with the corresponding model weights. Each of three fresh processes performs ten warmup passes and twenty measured passes. Inputs and all weight representations are materialized before timing; correctness checks are outside the timed region. The probe is a release build and excludes attention, norms, token dependencies, the tied Q6_K head, and sampling. [Operator measurements](./operators.json) and [the runner](./operators.py) preserve every timing.

| Q4 projection representation | Median pass time |
| --- | ---: |
| Original repeated metadata conversion | 24.363 ms |
| FP32 sidecars prepared once | 16.370 ms |
| Packed Q4 plus original scales, implicit symmetric bias | 12.343 ms |

The new path streams 5,449,973,760 bytes of packed Q4 and 681,246,720 bytes of scales per full projection pass. Its effective logical bandwidth is **496.7 GB/s**, or **80.9%** of Apple's advertised 614 GB/s peak for this M5 Max configuration. This is logical tensor traffic divided by wall time, not a DRAM-counter measurement. The original path additionally reads FP16 sidecars, writes converted FP32 sidecars, and reads them again for QMM; counting those intermediate transfers gives approximately twice the new Q4 traffic. The probe's simpler `logicalGBps` field for the original variant excludes those conversion transfers, so that column must not be treated as a directly comparable hardware utilization figure. [Apple specifications](https://www.apple.com/macbook-pro/specs/)

An optimistic full-model bandwidth bound includes the Q6_K head and one unique K/V read, assumes the full advertised bandwidth is sustained, and assigns zero cost to arithmetic, synchronization, norms, dispatch, or sampling. It is about 82.7, 77.1, and 73.2 tokens/s at the three measured contexts. Actual generation is below this bound. Attention compute/reduction, repeated cache reads, host scheduling, the output head, and non-matmul operations remain. Reaching roughly four-fifths of peak in the Q4 projection probe does not mean the complete model reaches four-fifths of its theoretical ceiling.

The speedup has a memory tradeoff. FP32 prefill sidecars add roughly 1.269 GiB relative to the original cache representation; retaining original Q4_0 scales for decode adds another 0.634 GiB. The new configuration therefore retains about 1.903 GiB more quantization metadata. Sliding read compaction reduces temporary attention memory substantially at long contexts. Per-run MLX allocator snapshots in the results track active arrays and temporary peaks; they exclude the private paged-KV pools and process RSS, so they are not total system-memory measurements. Lower-memory machines were not tested here.

## Correctness and validation

The mixed QMV tests cover short/tail tiles, real projection dimensions, FP16 and FP32 sidecars, explicit and implicit biases, and non-BF16-representable scales. On the captured real projections, changed elements remain within one BF16 ULP and the relative L2 error is reported in the operator artifact. This preserves scale precision while allowing the small reduction-order differences of a different kernel.

GPU regressions compare all supported D512 head layouts and all legal stripe counts against the generic attention implementation with nonuniform scores/values and a partial last page. Additional tests check compiled-primitive identity, invalid partition plans, absolute sliding-cache positions, partial pages, masks, symmetric-zero-point overrides, and planner behavior under different timing curves and noise. The planner's simulated latency observations are unit tests, not fabricated model-benchmark inputs.

The final validation record is in `results.json`: Gemma and GGUF Rust suites, the broader Rust library suite with the previously identified unrelated Qwen INT8 tests excluded, paged primitive integration checks, clippy, native release build, and 34 focused TypeScript tests including real Gemma GGUF text/image/audio loading and inference. Local checks are not remote CI. The numerical kernels and complete replay have only been measured on the listed M5 Max; other machines use capability checks and their own timing observations, but need physical-device validation.

Output hashes differ across implementations and can also differ across adaptive runs as the reduction plan changes. The result file reports this directly. Equal workload lengths and bounded operator error do not establish equal answer quality. No quality-parity or bitwise-parity claim is made. A 256-token continuation can end during thinking; these timings do not measure completion of the entire historical review task. The failed GPU trace collection was abandoned and supplies no evidence.

## Reproduction and related evidence

Run `oxnode docs/research/gemma4-optimization-2026-09-10/benchmark.ts run` after building the addon and restoring the locally retained real fixture. Then run `python3 docs/research/gemma4-optimization-2026-09-10/summarize.py`. The source history and generated text remain private under `.cache/benchmarks`; the public artifacts contain provenance, numeric measurements, and hashes. The original benchmark documents reconstruction of the production system wrapper because the historical session did not record that wrapper.

`MLX_GEMMA4_DECODE_TUNING=0` disables learning; `MLX_GEMMA4_MIXED_QMV=0` restores the stock decode QMM path. Existing route/stripe overrides and `MLX_GEMMA4_DECODE_EARLY_EVAL_LAYERS` remain diagnostic controls. No such performance override was used in the final matrix. The fixed 512-token prefill chunks match the original comparison protocol; they are not a newly fitted production default.

[Issue #142](https://github.com/mlx-node/mlx-node/issues/142) concerns a different Qwen architecture and workload. Its hybrid linear/full-attention mix, quantization, cache path, and context lengths explain why its throughput ranking cannot predict this Gemma result. The linked path audit compares the implementations and distinguishes later Qwen changes from the commit used by the issue benchmark.
