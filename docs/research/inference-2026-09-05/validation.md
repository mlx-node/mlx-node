# Inference validation

Validated on Apple M5 Max, 128 GiB unified memory, macOS 26.6.2.
The final inference implementation is `b7ff7f23`; subsequent cleanup only trims
research artifacts. [Measured results and decisions](followup.md) distinguish
whole-model comparisons from operation benchmarks and rejected candidates.

## Local checks

| Gate                                | Result                    | Coverage and limits                                                                             |
| ----------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| Final Rust core debug suite         | 3,422 passed, 110 ignored | Strict Metal, serial; 18 int8 cases excluded                                                    |
| Final TypeScript suite              | 3,294 passed, 38 skipped  | 182 passing files, 17 skipped; retained native binding                                          |
| FP32 finite-difference cases        | 2 passed                  | Explicit rerun with TF32 disabled                                                               |
| Original-verifier numerical oracles | Dense and MoE passed      | Both MoE projection modes; unequal prompt positions; mixed widths; full/partial/zero commit     |
| Shared transaction faults           | 6 passed                  | Recording, rollback, target commit, completion and publication isolation                        |
| MTP runtime isolation               | Dense and MoE passed      | Greedy/sampled cancellation, peer isolation and history/frontier checks                         |
| Real-model warm continuations       | Both families passed      | Two owners reused 119 and 116 tokens, then sampled MTP smokes completed                         |
| SSD process restart                 | Both families passed      | Two concurrent owners each reused 400 tokens and installed both recurrent sidecars              |
| Build and static checks             | Passed                    | Native build, all-target Clippy, Rust formatting, TypeScript and standalone benchmark typecheck |

The SSD restart checks use separate capture/restore processes and isolated cache
roots. Dense restored 26,487,150 validated bytes; MoE restored 16,707,950.
Drains completed with zero write errors, queue drops and corruptions.

Earlier gates at `4bf6b8b1`: 8 release int8 tests (10 ignored), 109 core
integrations (133 ignored), 336 paged/cache tests (17 ignored), four explicitly
selected paged MTP tests, and real Qwen serial/uniform/ragged/interleaved
concurrency. Those counts describe that checkpoint, not a repeat on the final
revision. Remote status is reported by [PR #138 checks](https://github.com/mlx-node/mlx-node/pull/138/checks).

The debug suite's two finite-difference cases return early in default TF32 mode;
the explicit FP32 rerun covers them. An optional Gemma mobile checkpoint is
absent, so its pre-existing repack test returns early. Ignored or unavailable
checkpoint cases are not claimed as executed coverage. Earlier commands used
`MLX_REQUIRE_METAL`; final GPU gates use the correct `MLX_TEST_REQUIRE_METAL`.

An unseeded Nemotron fixture was seeded after exposing order-dependent synthetic
logits. Random recurrent MTP runtime fixtures showed late greedy differences
across batch shapes. Cancellation/RNG fixtures now use a fixed non-uniform head
with the full attention/recurrent stack; separate original-verifier oracles
retain nonconstant weights. These tests establish isolation and bounded numerical
agreement, not universal token-for-token batch invariance. Real-model output
differences and rejected performance candidates are reported in the follow-up.

A bounded independent source audit found no confirmed defects in committed
history alignment, draft KV frontiers, per-owner tape replay, SSD publication
or admission reservations. Source review is separate from runtime validation.

## Reproduction and provenance

GPU tests and performance runs were serialized. Alternating model comparisons
exclude builds, other GPU tests and trace export; warmups are excluded. Raw JSON
outputs, full logs and immutable bindings are retained outside the repository.
The scripts below accept explicit binding/model/output paths and emit new JSON;
their checked-in sources and summary tables are the reviewable evidence.

The retained scheduled-MTP addon SHA-256 is
`900c7817ef1a9ade7f065de1b685715badeacc318d3c8282b4d6dc001bf93a8d`;
its baseline at `4bf6b8b1` is
`6acdcf45c702d9b4aeb94e9e93adda2586c11b92ce89c0e0112d3b25c6b54a75`.
The final Gemma timing binding is
`20e526e4a6c8aed82d1ffaac86f134e493dbe9560584fd5e20332287feac897b`.
Later source changes to the measured MTP binding add tests/docs, collapse a
nested frontier guard for Clippy, and rename a retained-hidden parameter;
they do not change measured inference behavior.

```sh
MLX_TEST_REQUIRE_METAL=1 cargo test -p mlx-core --lib -- \
  --nocapture --skip models::qwen3_5::int8_gemm::tests --test-threads=1

MLX_CONTINUOUS_BATCHING=1 MLX_PERSIST_PAGED_CACHE=0 MLX_AGENT_METRICS=0 \
  oxnode docs/research/inference-2026-09-05/benchmarks/mtp-concurrent.ts \
  /absolute/binding/index.cjs /absolute/model /tmp/mtp.json revision-label Qwen35 1,2 2
```

Use `Qwen35Moe` for the MoE workload. [mtp-continuation.ts](benchmarks/mtp-continuation.ts)
checks warm continuation and sampled requests. [mtp-ssd-restart.ts](benchmarks/mtp-ssd-restart.ts)
requires `MLX_PERSIST_PAGED_CACHE=1` and an isolated `MLX_COLD_CACHE_DIR`; run
`capture`, then `restore` in separate processes against the same root.
[dspark-concurrent.ts](benchmarks/dspark-concurrent.ts) reproduces fixed/adaptive
Gemma comparisons. The remaining scripts cover ordinary AR, prefix reuse,
host exports and the earlier singleton speculative baselines.
