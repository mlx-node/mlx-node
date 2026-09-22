# Portable Qwen dense prefill: M3 Pro investigation

The supplied September 17 trace exposes a missing fast path for D=256 full
attention on pre-M5 GPUs. Prefix reuse works, but long-context suffix processing
falls back to either a materialized score matrix or a paged kernel designed
around individual queries. The latter repeatedly reads the context and retains
large per-query, per-partition outputs. This change adds ordinary SIMD-group
matrix attention for that geometry and tells the memory planner about it.

## Evidence from the supplied run

The session identifies `Qwen3.8-27B-UD-Q4_K_XL`; the user identifies an M3 Pro
with 36 GB. The logs do not include a software revision, thermal telemetry, or
swap measurements. The attached conversation was treated as data. No tool
commands or instructions inside it were executed.

`m3-pro-log-summary.json` preserves input hashes, all 43 profile summaries, and
memory/route aggregates without conversation content.

| Observation                                 | Result                                                                    |
| ------------------------------------------- | ------------------------------------------------------------------------- |
| Initial 4,180-token prompt                  | 50.986 s prefill; 51.121 s TTFT; 8.376 tok/s MTP decode                   |
| 79,038-token conversation, 1,026 new tokens | 81.058 s prefill; 5.813 tok/s MTP decode                                  |
| Prefix state                                | 40 live, 3 cold; all 16 logged MTP continuations reused their live prefix |
| Attention route decisions                   | 74 paged-varlen, 30 gathered SDPA; D256 full-SDPA capability false        |
| GPU recommended working set                 | 28,753.922 MiB                                                            |
| Paged pool allocation                       | 5,587 MiB throughout                                                      |
| Reported GPU headroom                       | 707.945–5,829.144 MiB                                                     |
| Total measured prefill                      | 4,474.997 s                                                               |
| Total measured decode                       | 4,075.513 s                                                               |
| Median decode                               | MTP 6.188 tok/s; AR 5.568 tok/s, on different turns                       |

The 1,026-token continuation contains a 1,012-token materialized chunk and a
14-token tail. For that first chunk, the trace estimates 4,389.653 MiB for
unfused SDPA versus 1,942.784 MiB for varlen, with 3,318.908 MiB headroom. The
new portable estimate is 705.094 MiB, including the conservative gather-copy
allowance and 64 MiB fixed overhead. A regression replays these exact inputs
through the production planner and verifies the route changes to SDPA.

The roughly 5–8 tok/s decode rate is a separate issue from these prefill stalls.
M3 Pro has [150 GB/s specified memory bandwidth](https://support.apple.com/en-ie/117737).
Streaming a roughly 17.6 GB dense quantized checkpoint already imposes a much
lower throughput ceiling than on a Max device. This is a bandwidth-based
explanation, not a measured attribution of every decode millisecond. The log
does not establish swap as the cause. MTP and AR figures above are not a paired
MTP speedup comparison.

## Implementation

`mlx_portable_sdpa.cpp` instantiates the pinned MLX Steel attention template
with BQ=32, BK=16, D=256 and four SIMD groups. BF16/FP16 scratch is 29,184 bytes
per threadgroup, below the 32 KiB class of earlier Apple GPUs. These are tile
dimensions, not assumptions about GPU core count or installed RAM. The runtime
checks each compiled pipeline's thread width, thread limit and shared-memory
requirement against the actual device before advertising support.

The online softmax retains no sequence-sized score matrix and no per-query
partition-output tensor. Ragged Q/K lengths use the template's bounded loaders;
the true `key_length - query_length` causal offset is preserved. Q/K/V strides,
GQA and batches are retained. The kernel runs on MLX's command encoder so paged
write/gather dependencies remain ordered.

The preamble is generated from the linked vendor source at build time and
embedded in the native library. Runtime compilation needs neither a local MLX
checkout nor optional MLX JIT exports. The private preamble also backports the
two `BD >= 128` V-tile synchronization guards from upstream PR #4185; merely
instantiating the pinned template would omit these barriers at D256. The build
checks both patch sites and fails if a vendor update needs review. The vendor
submodule is unchanged.

Automatic use is limited to Metal without NAX, identical BF16/FP16 operands,
D=256, causal inference, more than eight query tokens, and supported layouts.
Other precision, masks, training, CPU/CUDA, vector decode and M5's existing
automatic route retain their previous implementation. A failed capability
probe retains the existing fallback and reports the reason once.

The Qwen paged planner now distinguishes portable fusion from NAX fusion.
Portable fusion applies below NAX's 1,024-query threshold and needs no NAX
padding buffers. Estimates remain conservative for unsupported precision and
other head sizes. No model cache or context limit is raised.

## Audit of existing policy

| Policy                                  | Finding / action                                                                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| D256 NAX-only full-attention gate       | Confirmed missing portable route; fixed here without pretending NAX exists on M3                                                            |
| 2,048-token launcher chunk              | Existing mlx-lm-sized default; reducing it globally does not remove repeated context scans. Retained pending paired device measurements     |
| Eight-layer prefill evaluation interval | Bounds activation lifetimes. Retained; the trace does not demonstrate this is the root cause                                                |
| Fixed 2 GiB SDPA reserve                | Can be restrictive on smaller devices; retained for now because the new lower estimate fits the recorded case without weakening the reserve |
| GDN chunked-kernel generation gate      | Already removed in current source; per-step is default on every generation                                                                  |
| MTP chained scheduling gate             | Already off by default before M5; not incorrectly forced on the reported M3                                                                 |
| Wired memory / paged headroom           | Already queries the device working set and external Metal pool; the supplied data does not justify raising memory limits                    |
| Prefix invalidation                     | Not the recurring stall: 40/43 live prefixes; cold post-compaction turns legitimately require prefill                                       |

Source baseline: repository `377e78f3`, MLX submodule
`d957e668b0a033840b60d2f5cd9810bb96f0ebb4`. The current primary checkout had
uncommitted inference work; this investigation and implementation use an
isolated checkout and do not incorporate that unfinished work.

Upstream research corroborates the mechanism: [MLX PR #3293](https://github.com/ml-explore/mlx/pull/3293)
proposed instantiating the existing Steel template for D256 and reported
long-context validation on M2 Ultra. It was closed, not merged. [Issue #3658](https://github.com/ml-explore/mlx/issues/3658)
describes the same score-matrix memory growth on a 36 GB setup despite chunked
prefill. Those reports support investigating this path; they are not performance
measurements of this patch or of the user's device. The delegated investigation
also identified the merged successor [PR #4185](https://github.com/ml-explore/mlx/pull/4185),
which closed #3658 and exposes the memory/throughput tradeoff through an
explicit `force_fused` option. That API and its barrier fix are absent from our
pinned fork. This patch provides the narrow D256 bridge and backports those
barriers without replacing the fork's other kernels.

## Validation and reproduction

Measurements below were taken on **M5 Max, 128 GB**, forcing the portable
kernel. They demonstrate the implementation and its memory behavior; they do
not establish M3 throughput. Compilation finished before the final timing
samples. GPU benchmarks and model runs were serialized on the shared desktop;
thermal conditions were not controlled.

For the recorded 79,024-context / 1,012-query shape, two fresh processes per
route, each with one warmup and three completed samples, measured:

| Production gather + attention | Median time per process | MLX allocator peak |
| ----------------------------- | ----------------------- | ------------------ |
| Existing paged varlen         | 1,122–1,211 ms          | 1,891.3 MiB        |
| Portable Steel SDPA           | 221–231 ms              | 641.7 MiB          |

This is approximately **5.2x faster for this attention operator**, with 66%
less MLX allocator peak memory. The peak excludes the external Metal paged
pool, which has the same geometry in both arms. Full samples and capability
checks are in `operator-results.json`; this is not a model-wide speedup.

The smaller shape sweep is deliberately retained in `shape-sweep-results.json`.
Portable attention was not universally faster than unfused SDPA: at 32,768
context / 67 queries, it took 18.9 ms versus 13.3 ms, while using less peak
memory. At that shape varlen took 28.9 ms. The automatic M5 route is unchanged;
these forced-path measurements cannot justify a new M3 query cutoff.

The full 27B public review fixture (18,718 initial prompt tokens) completed
both turns with identical input and output hashes across portable and
old-route controls, exactly 64 generated
tokens per turn, and 18,781 cached tokens on the continuation. On the final
build after compilation finished, initial TTFT was 52.59 s versus 91.43 s;
continuation TTFT was 3.20 s versus 7.66 s. Decode timing varied from 6.17 to
9.10 tok/s between phases
and runs. These single-pair timings are retained in `session-results.json`
with the native addon's hash for transparency and correctness evidence,
**not a stable end-to-end speedup claim** on this shared machine.
The old control forces varlen continuations and unfused SDPA; it does not
recreate M3 memory pressure or disable unrelated M5 kernels.

Numerical coverage forces this exact portable kernel on any Metal GPU and
compares to stock MLX with an explicit causal mask. It covers FP16/BF16,
square/rectangular and ragged/aligned shapes, batches, transposed strides,
split cached continuations, nonconstant paged writes/gathers, unsupported FP32,
and rollback. Comparisons require finite outputs and bounded absolute and
relative errors; BF16 reduction ordering need not preserve every greedy tie.

Validation passed: the canonical native build and packaged-metallib checks,
25 Qwen attention/planner tests, both explicit GPU tests (39 numerical
comparisons), Clippy with warnings denied, Rust formatting, benchmark-script
lint, and diff whitespace checks. `validation-results.json` records the
individual numerical errors and validation boundaries. Actual M3 hardware,
remote CI, and an MTP end-to-end A/B were not available in this run.

```sh
cargo test -p mlx-core --release --test portable_d256_sdpa -- --ignored --nocapture --test-threads=1
```

The existing paged-prefill operator benchmark now has an explicit `portable`
expectation and fails before pool allocation if the requested kernel is
unavailable. Its timings include the production pool gather and completed GPU
evaluation, not just graph construction. The benchmark's Q=K=0/V=1 output
check is supplemented by the nonconstant numerical tests above.

```sh
MLX_PORTABLE_D256_SDPA=1 \
MLX_QWEN35_PREFILL_BENCH_CONTEXT=79024 \
MLX_QWEN35_PREFILL_BENCH_QUERY=1012 \
MLX_QWEN35_PREFILL_BENCH_ROUTE=sdpa \
MLX_QWEN35_PREFILL_BENCH_EXPECT=portable \
MLX_QWEN35_PREFILL_BENCH_WARMUP=1 \
MLX_QWEN35_PREFILL_BENCH_ITERS=3 \
cargo test -p mlx-core --release --test qwen35_paged_prefill_operator_bench \
  qwen35_paged_prefill_operator_benchmark -- --ignored --exact --nocapture
```

Run one process at a time. For the recorded old route, change route/expectation
to `varlen` and set `MLX_PORTABLE_D256_SDPA=0`. For the score-matrix control, use
`sdpa`/`fallback`, `MLX_PORTABLE_D256_SDPA=0`, and
`MLX_ENABLE_D256_FULL_SDPA=0`. Environment switches are process-cached.

For an application A/B, rebuild using `yarn build:native`, then repeat the same
saved conversation with automatic settings and with `MLX_PORTABLE_D256_SDPA=0`
in fresh processes. Keep model, input, context, MTP settings, memory limits,
power mode, and thermal conditions identical. Inspect `portable_d256_available`
and `estimated_sdpa_mib` in the inference trace. This does not claim M3
end-to-end speed until that device has completed a matched run.

`benchmark-session.ts` exercises the pinned public Qwen review fixture and a
real cached continuation with 64 greedy AR tokens per turn. It verifies the
fixture hash, records input/output hashes and prefix reuse, and refuses an
unexpected checkpoint size or insufficient physical capacity before loading.
The tested checkpoint's full SHA-256 was separately checked against the
fixture manifest. Neither the model directory nor the attached private session
is modified. Run it with `oxnode` from the repository root:

```sh
MLX_PORTABLE_D256_SDPA=1 MLX_PAGED_PREFILL_CHUNK_SIZE=2048 \
oxnode docs/research/portable-inference/benchmark-session.ts \
  /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf portable-session.json 16k
```
