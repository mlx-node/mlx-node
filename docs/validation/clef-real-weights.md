# CLEF released-weight acceptance

Validated on 2026-10-10 on an Apple M5 Max with 128 GiB unified memory.
Both checkpoints were downloaded using the repository CLI:

```sh
yarn mlx download model --model Cloudflare/clef-flash --output ~/.mlx-node/models/clef-flash
yarn mlx download model --model Cloudflare/clef --output ~/.mlx-node/models/clef
```

The downloader completed full-manifest verification. Native and reference
inference ran sequentially, with one model resident per test process.

| Checkpoint         | Revision                                   | Cases | Compared probabilities | Maximum absolute probability difference | MLX allocator peak |
| ------------------ | ------------------------------------------ | ----: | ---------------------: | --------------------------------------: | -----------------: |
| CLEF Flash 9B BF16 | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` |     6 |                     28 |                             0.000785593 |           28.51 GB |
| CLEF 27B BF16      | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |     6 |                     28 |                             0.002309263 |           88.52 GB |

All 13 modal decisions per model agreed with the independent reference. Every
input token count matched. The comparison checks a maximum absolute probability
difference below 0.01 for BF16 arithmetic; the table reports the actual measured
differences. Full answers and per-case results are in
[Flash results](clef-flash-bf16.json) and [CLEF results](clef-bf16.json).

## Coverage

The shared [acceptance inputs](../../__test__/fixtures/clef/acceptance.json) cover
`noul`, `choice`, and `score`; positive and negative decisions; structured JSON;
French and Chinese text; integer-looking question IDs; a single-option choice;
and a 2,960-token state. These are functional acceptance cases, not an accuracy
benchmark.

Both checkpoints passed:

- Normal host discovery and loading, without creating chat sessions.
- Every declared semantic expectation in all six inputs.
- Exact repeat results after intervening requests.
- Cancellation during a long request, followed by exact recovery on the first input.
- Authenticated `/v1/systemone`, request IDs, and explicit media rejection.
- Real TypeSafe SDK 0.6.0 requests and model listing.

The native acceptance runner is [verify-release.ts](../../scripts/clef/verify-release.ts).
The reference comparison is [compare-release.ts](../../scripts/clef/compare-release.ts).
For example, from the repository root:

```sh
oxnode scripts/clef/verify-release.ts ~/.mlx-node/models/clef-flash /tmp/flash-native.json /path/to/typesafe-sdk/dist/index.mjs
oxnode scripts/clef/compare-release.ts /tmp/flash-native.json /tmp/flash-reference.json /tmp/flash-comparison.json
```

## Reference and timing conditions

The reference used PyTorch 2.14.1, Transformers 5.19.0, MPS, BF16, eager attention,
and the released `JointSchemaHead`, encoder, batching, and `ClefModel` code.
Its text-only adapter loads the same conditional-generation backbone and output
embeddings, disables KV reuse, and applies a per-question FP32 softmax. It uses
the tokenizer directly, so no image processor is needed. Reference scripts live
outside the repository; no Python files are part of this change.

The released Python implementation is identical at both pinned model revisions:
SHA-256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
The reference used PyTorch's fallback recurrent kernels; its timings are not
used for a speed comparison.

Flash's recorded load took 10.57 seconds on a repeat run. Its five short cases
took 87–340 ms, and its long case took 1.08 seconds. CLEF 27B's first native load
after download took 137.58 seconds. Its first request took 23.43 seconds; subsequent
short cases took 297–1,174 ms, and the long case took 3.89 seconds.

These are individual acceptance measurements, not a controlled benchmark. The
27B run had substantial system swap in use (about 26 GB observed after the native
run). Filesystem-cache state differed between runs. Allocator peaks describe MLX
buffers and do not represent total system memory use.

Real-checkpoint quantization quality, wider accuracy evaluation, throughput under
load, and media inference remain outside this acceptance run.

## Benchmark rerun on October 10, 2026

The rerun used the same native binary, checkpoint revisions, and six acceptance
inputs, with one fresh process per model (Flash followed by CLEF). All answers,
probabilities, and token counts matched the previous native results exactly.
HTTP, TypeSafe SDK, repeat, and cancellation recovery checks passed again.
The independent reference outputs were reused, not regenerated.

| Model      | Short-case median, previous → rerun | 2,960-token case, previous → rerun | Rerun load | Peak MLX allocation |
| ---------- | ----------------------------------- | ---------------------------------- | ---------- | ------------------- |
| CLEF Flash | 151 → 139 ms                        | 1.08 → 1.67 s                      | 41.65 s    | 28.51 GB            |
| CLEF 27B   | 696 → 1,691 ms                      | 3.89 → 5.80 s                      | 176.11 s   | 88.52 GB            |

The short-case median covers five different inputs of 120–340 tokens, including
the first inference; each case has only one timed sample. System swap usage was
18,031 MiB before the run, reached 33,312 MiB while loading CLEF, and ended at
29,807 MiB. These measurements do not establish a controlled performance
regression. Model loading is measured separately from inference.

The [rerun evidence](clef-benchmark-rerun.json) includes individual case timings,
checkpoint revisions, native binary and fixture hashes, and parity checks.
