# Gemma 4 GGUF: performance findings and safeguards

Essential findings from [PR #144](https://github.com/mlx-node/mlx-node/pull/144), measured September 10, 2026 for `gemma-4-12b-it-qat-q4_0.gguf`. Full scripts and samples remain in [Git history at `f193ade2`](https://github.com/mlx-node/mlx-node/tree/f193ade24e02c3ab0abba5aab74e1c916e74f454/docs/research).

## What caused the gap

The checkpoint has 328 Q4_0 projections and a Q6_K tied embedding/output head. Its 48 layers comprise 40 sliding-attention layers (16Q/8KV, D256, window 1,024) and eight global layers (16Q/1KV, D512).

| Cause | Implemented correction | Source |
| --- | --- | --- |
| Affine QMM promoted BF16 activations plus FP16 scale/bias arrays to FP32, rebuilding large metadata casts on each forward. Missing nested text dtype also disabled the existing load-time hoist. | Record the actual nested BF16 text dtype in the native cache; hoist prefill metadata once. Single-row Q4 decode reads original scales directly, derives a symmetric bias in registers, accumulates in FP32, and returns BF16. | [Persistence](../../crates/mlx-core/src/models/gemma4/persistence.rs), [Q4 Metal bridge](../../crates/mlx-sys/src/mlx_affine_qmv.cpp) |
| Sliding attention retained full logical history in its read view, partitioned over it, and masked old tokens after loading keys. Prefill gathered the full history too. | Trim retired whole pages from the private decode view; gather only window plus current prefill chunk. For this fixture the gather is bounded by 1,536 tokens. | [Paged cache adapter](../../crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs) |
| Conservative global-attention routing and CPU submission left supported faster paths unused. | Calibrate attention partitions and early submission from completed real decode tokens on the loaded device. | [Decode tuning](../../crates/mlx-core/src/models/gemma4/decode_tuning.rs) |

This was **not a whole-weight Q4 → BF16 → Q4 cycle**. Weights stayed packed; embedding lookup dequantizes selected rows, and the tied output head uses packed matmul.

Qwen's result in [issue #142](https://github.com/mlx-node/mlx-node/issues/142) cannot predict this Gemma result: the measured Qwen checkpoint predominantly uses K/IQ quantization, hybrid linear/full attention, D256 global heads, and a different benchmark protocol. K-quant metadata does not follow the same affine dtype-promotion path.

## Recorded benchmark

Medians of three fresh-process runs per runtime and context on an M5 Max, 40 GPU cores, 128 GB, macOS 26.6.2. The baseline MLX measurements were earlier that day; llama.cpp below was freshly measured alongside optimized MLX.

| Input tokens | Original MLX decode tok/s | Optimized decode tok/s MLX / llama.cpp | Prefill tok/s MLX / llama.cpp | Request seconds MLX / llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 7,733 | 28.21 | 50.53 / 48.70 | 1,153.2 / 1,252.0 | 11.76 / 11.24 |
| 40,528 | 16.15 | 39.84 / 35.29 | 1,000.4 / 740.9 | 46.99 / 61.93 |
| 66,904 | 12.22 | 34.05 / 28.80 | 872.1 / 526.5 | 84.40 / 135.93 |

Decode improved 1.79–2.79× over original MLX; the shortest request remains slower than llama.cpp overall. Three repeats on one active desktop do not establish statistical confidence or performance on other devices. No competing inference/compilation or thermal warning was recorded.

**Workload:** real mlx-node session `01a067cf-b782-7b58-855e-8158dcb283ab`, reviewing Oxc PR #745, at complete turn boundaries. It originally used Qwen; the absent system prompt was reconstructed using Pi 0.84.4 and production message conversion. No padding, fabricated messages, or tool execution.

**Protocol:** identical token IDs, BF16 KV, greedy AR, high thinking, no speculation, 256 output tokens, zero prompt-cache hits, 512-token physical prefill chunks. Each process generated 32 warmup tokens from the shortest fixture, then reset its prompt cache. Adaptive state survives reset; calibration during the request counts toward latency. Runtimes alternated with 20-second cooldowns. Loading/warmup are excluded; disk artifacts were warm.

## Rules that must survive future optimization

- **Preserve precision and encoding.** Q4_0 differs from Q4_K. Never round FP16 scales to BF16. Derive `-8 × scale` only for metadata-authorized symmetry, preserving FP16 rounding; asymmetric overrides retain explicit biases. Q6_K's FP16 `.biases` is a super-block scale, not an additive bias.
- **Keep kernel guards.** Mixed QMV requires Metal, affine Q4/group32, one BF16 row, packed U32 weights, `K % 32 == 0`, `N % 8 == 0`, and compatible FP16/FP32 metadata. Prefill, other formats/shapes, and gradient transforms retain existing paths.
- **Preserve cache semantics.** Private read rebasing must retain absolute RoPE positions, ownership, prefix reuse, partial pages, and causal/window masks. Keep the masked boundary key needed for tile alignment.
- **Capture GPU plans.** Stripe counts belong in the graph-native primitive and compile identity, with capability/layout/scratch guards. Asynchronous execution must not read mutable thread-local policy.
- **Match loading and discovery.** Require sibling `config.json` and `tokenizer.json` files for root-level and nested Gemma GGUFs. Discover/convert projectors only for parsed unified vision/audio capabilities. Plain Gemma legacy audio/SigLIP settings and text-only unified configs must ignore irrelevant projectors, including ambiguous/invalid files. Unified audio remains supported with its companion.

Scope is existing GGUF loading/inference, including Q3_K/Q4_K/Q5_K/Q6_K, not K-quant creation. Cache identity includes relevant source/assets/media and corrected text dtype; SafeTensors loading is unchanged.

## Adaptive selection and rollback

Single-row Gemma paged AR searches supported power-of-two partitions, layer-count-bounded submission depths, then attention again. Each candidate gets one unscored warmup and three timed steps, reversing alternate rounds. A median win must exceed twice the observed median absolute deviation and 1%.

Existing token-completion timing includes host work; no extra forwards or GPU waits are added. Plans use an eight-entry model-local cache of power-of-two context scales, without machine-name presets. Batched/speculative policies are unchanged. This finite search neither proves global optimality nor continuously follows thermal/load changes.

Set `MLX_GEMMA4_DECODE_TUNING=0` to disable learning or `MLX_GEMMA4_MIXED_QMV=0` to restore stock decode QMM. Diagnostic route, stripe, and early-layer overrides take precedence; see [performance controls](../perf.md). Do not turn recorded winning settings into universal presets.

## Memory, correctness, and remaining headroom

The captured-activation Q4 probe measured **24.363 → 16.370 → 12.343 ms** for repeated casts → hoisted FP32 metadata → original scales/implicit bias. Each variant used three release processes, ten warmup and twenty timed passes over 328 projections, with operands resident and correctness outside timing.

The new path's 6,131,220,480 logical bytes/pass imply **496.7 GB/s**, approximately **81%** of advertised 614 GB/s. This is logical traffic/wall time, not a DRAM counter or full-model utilization. Original conversion intermediates add traffic beyond a weight-byte count. [Hardware specifications](https://www.apple.com/macbook-pro/specs/)

Including the Q6_K head and one unique KV read gives optimistic bandwidth-only bounds of **82.7 / 77.1 / 73.2 tok/s**. These assume peak bandwidth and zero compute/dispatch/synchronization cost. Attention reductions, repeated KV reads, the output head, and host scheduling remain measurement targets.

Memory increases **about 1.9 GiB**: 1.269 GiB of widened prefill metadata plus 0.634 GiB of original decode scales. Window compaction reduces temporary memory. MLX snapshots exclude private KV pools/RSS; smaller-memory devices remain unvalidated.

Projection differences stayed within one BF16 ULP (relative L2 approximately `9.56e-6`). Greedy outputs differ across runtimes and adaptive runs: no bitwise/answer-quality parity is established. A 256-token cap can end during thinking, so this is not completed-review latency. Numerical/attention regressions and real text/image/audio tests passed; broader local Rust checks excluded 18 unrelated Qwen INT8 tests. Failed GPU tracing supplied no evidence.

## Evidence anchors

| Artifact | Pinned identity |
| --- | --- |
| Model SHA-256 | `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b` |
| MLX library commit | `6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb` (unchanged submodule) |
| llama.cpp commit | `a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92` (build 10610, Metal/Accelerate) |
| Measured source patch SHA-256 | `df19f7b76b8b1e02067012667bb464725bb55048f91447cb87251568ba75f6f4` (base `185bc1b0`, including new source files) |
| Measured native addon SHA-256 | `2f380276a3fa2e3dc39f3e5ff377bc213152554d34d2ba3d819db733d3adface` |

Archived `gemma4-optimization-2026-09-10/results.json` holds all 18 samples, source/output hashes, and plans; runners and baseline/audit evidence share the Git-history link above. Later formatting/preflight/discovery changes leave the timed inference path unchanged; the matrix was not rerun.

## Pinned fixture

[`gemma4-oxc-review-v1`](../../scripts/fixtures/gemma4-oxc-review-v1.json) pins the exact messages, tool definitions, rendered prompts, and token IDs used above. The 362 KB compressed object is stored in the private Cloudflare R2 bucket `mlx-node-benchmarks`, under a versioned key containing the payload SHA-256. The manifest records the object and payload hashes, model/tokenizer identities, protocol, and archived runner identity. Public access is disabled; no completed-object expiration rule is configured. Private messages, outputs, and activations stay outside Git.

With Wrangler authenticated to the account in the manifest, restore and verify the fixture:

```sh
oxnode scripts/benchmark-fixture.ts fetch
oxnode scripts/benchmark-fixture.ts verify --tokenizer .cache/models/gemma-4-12b-it-qat-q4_0-gguf/tokenizer.json
```

`fetch` verifies the compressed object, decoded file, and all three token-ID hashes before saving to `.cache/benchmarks/gemma4-agent-2026-09-10/inputs.json`. An existing differing file is rejected. `verify` is offline; its optional tokenizer check detects current template/tokenizer drift. The manifest and hashes are the pin; storage administrators can still replace objects, so never bypass verification or overwrite this version. New messages, tools, or templates require a new fixture version.

For inference replay, retrieve the archived runner identified by the manifest and adapt its local repository/model/llama-server paths; use `oxnode` for TypeScript. Do not rebuild the fixture wrapper, whose dates/tools can change. Keep model and token IDs, caches, output lengths, and timing policy matched; record contention and source/binary hashes. The recorded llama.cpp six-thread setting is a comparison parameter, not an optimum for every machine. A fixture pins the workload, not expected speed or generated answers.
