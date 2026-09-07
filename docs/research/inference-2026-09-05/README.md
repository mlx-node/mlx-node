# Inference architecture and performance reference

Design and measurements from 2026-09-05–06, implementation introduced at
`b7ff7f23` with admission fixes from the 2026-09-07 review.
Local performance results use an Apple M5 Max with 128 GiB unified memory and
macOS 26.6.2. Reference projects informed the design; they were not benchmarked
against mlx-node. Raw results and build artifacts are kept outside the repository.

## Model integration

Keep the registry, immutable execution plan, shared scheduler and family-owned
tensor/cache implementations. All seven chat families use
`ModelCommand<FamilyCommand>` and default chat-barrier dispatch. Families without
extra commands use `Infallible`; other families define only their extensions.
The command adapter macro and conversion/view traits are removed. The concrete
NAPI export macro remains because it generates binding methods.

Traits with defaults own reusable algorithms; family hooks supply model math
and state access. `ScheduledDraftVerify` owns the verification transaction.
`ScheduledMtpTarget` shares dense/MoE proposal chaining, draft history and recurrent
commit logic. New models still need their forward pass, batched tensor program
and auxiliary-state lifecycle. See [adding a model](../../adding-a-model.md) and
[the engine architecture](../../inference-architecture.md).

vLLM's useful separation is stable request ownership versus temporary execution
batches. Its CUDA graphs and multiprocessing topology are not prerequisites for
our model thread. [Model Runner V2](https://docs.vllm.ai/en/latest/design/model_runner_v2/).

## Speculative scheduling and cache invariants

Keep reserved capacity, optimistic verifier writes, accepted history and completed
GPU work separate. A rejected attention suffix rewinds its logical cursor;
recurrent GDN state additionally needs accepted-prefix replay. Publish reusable
cache state only after its committed frontier and GPU completion agree.
This matches vLLM's separation of lookahead capacity and finalized prefix tokens.
[KV cache manager](https://github.com/vllm-project/vllm/blob/874df9373dab532543a0229fb2f144f7c14093ae/vllm/v1/core/kv_cache_manager.py#L389-L562),
[recurrent state](https://github.com/vllm-project/vllm/blob/874df9373dab532543a0229fb2f144f7c14093ae/vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L89-L118).

| Path               | Execution policy                                                                                             | State that must remain request-owned                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Gemma DSpark       | Fixed-depth text verification shares scheduler waves; greedy adaptive requests use measured allocation costs | Draft context, RNG, sliding/full KV frontiers and accepted target taps              |
| Muse DFlash        | Scheduled verification requires `MLX_SCHEDULED_DFLASH=1`; default keeps the flat route                       | Draft KV and target-context provenance                                              |
| Qwen dense/MoE MTP | Scheduled fixed-depth MTP requires `MLX_CONTINUOUS_BATCHING=1`; adaptive MTP retains its whole-turn lane     | Normalized target seed, draft KV history, GDN snapshot and tape slices              |
| Qwen DFlash2       | Retains its flat whole-turn route                                                                            | Exact flat-cache provenance; paged AR state cannot be reused as flat verifier state |

Native MTP groups equal verifier widths so GDN sees `[N,T,H]`; only attention
packs query rows into paged kernels. Verification uses scratch recurrent caches.
Close attention tickets, replay each owner's accepted prefix (including full
acceptance), complete committed state together, then settle. Missing GDN tapes
fail closed. Failure or cancellation must not publish a failed owner's state or
abandon a healthy peer's transaction.

On a cold prompt, seed MTP history with `hidden(previous) → embedding(next)`.
The committed draft frontier is the consumed target frontier minus one. Trim
speculative draft KV before appending committed pairs. Cached-prefix starts,
including continuations and preemption replay, do not have complete draft history.
They decline scheduled MTP, release the draft reservation, and keep their target
prefix in the shared AR scheduler. Re-enable speculation only when complete
draft history can be restored. Draft caches are temporary request state charged
to admission, not a second cold RAM cache.

Adaptive Gemma costs are keyed by the ordered per-owner draft-length vector,
not total query count. Keep at most eight measured shapes and reset on owner
order, context band, cap or draft-context lifetime changes. A zero-draft probe
retains contexts for calibration; a measured AR decision releases them. Exclude
calibration-only work from the AR baseline. Fixed depth remains the default:
calibration overhead can outweigh adaptive gains. Adaptive barriers apply only
when a loaded speculative decoder admits the requested streaming shape; an
unsupported request that resolves to AR must not drain concurrent rows.
Confidence-dependent truncation is restricted to greedy requests because it can
change a sampled proposal's conditional distribution.
[vLLM adaptive verification](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/).

The upstream algorithms share infrastructure but preserve different proposal
semantics: MTP advances autoregressive draft positions; DFlash populates draft
context KV with accepted target rows and pads rejected/nonresident slots; DSpark
adds checkpoint-specific Markov sampling, confidence and anchor conventions.
[MTP](https://github.com/vllm-project/vllm/blob/874df9373dab532543a0229fb2f144f7c14093ae/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py),
[DFlash](https://github.com/vllm-project/vllm/blob/874df9373dab532543a0229fb2f144f7c14093ae/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L551-L640),
[DSpark](https://github.com/vllm-project/vllm/blob/874df9373dab532543a0229fb2f144f7c14093ae/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L40-L55).

Code: [shared transaction](../../../crates/mlx-core/src/engine/scheduled_verify.rs),
[scheduler](../../../crates/mlx-core/src/engine/hybrid_scheduler/speculative.rs),
[native MTP](../../../crates/mlx-core/src/models/qwen3_5/scheduled_mtp.rs),
[adaptive costs](../../../crates/mlx-core/src/engine/verification_budget.rs).

## Unified memory, transfers and SSD

Unified memory removes a separate device address space for shared allocations.
It does not remove owned vector copies, private-buffer staging, layout conversion
or GPU completion dependencies. `eval` is a completion boundary, not a copy;
host-pointer constructors copy into owned storage. Handle clones and array views
can retain their parent allocation, so compact retained state deliberately.
[MLX memory model](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html),
[lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

| Boundary              | Retained optimization or requirement                                                                                                                                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prefill attention     | Qwen3 uses direct paged attention for suffixes up to 16 tokens; larger chunks use graph-native gather plus SDPA. Avoid private-pool → CPU unpack → MLX reconstruction.                                                                                            |
| Prefill output        | Slice dense retained hidden tails before normalization. MoE projects only the last vocabulary row; retain normalized prompt rows only when committed MTP history needs them.                                                                                      |
| Paged metadata        | Reuse immutable one-token mappings across layers. Validate owner order, table incarnation, physical revision, frontier and pool generation. Scan already available slot metadata without a GPU max reduction; retain guards for lazy inputs.                      |
| Sampling and drafting | Group independent evaluation roots. Keep draft token-to-embedding chains, dense distributions and residual fallback selection on device; read compact IDs for host decisions. PaddleOCR reuses sampled IDs for embeddings and reads them once for output/history. |
| Host exports          | Evaluate the actual array; remove redundant add-zero work and materialize contiguous storage only when necessary. Requested tokens/log probabilities still require host reads.                                                                                    |
| SSD capture           | Complete private KV → bounded shared staging → immutable owned writer bytes. Sidecars must describe the exact published recurrent/sliding boundary.                                                                                                               |
| SSD restore           | Overlap bounded reads with batched uploads into reserved private slots. Checksums, identity, pool generation and completion proofs gate publication; charge staging memory once.                                                                                  |

Ordinary token reads support detokenization, stopping, penalties and protocol
guards. DFlash2's CPU selector reads compact top-k/edge tables. Media shape
decisions and output postprocessing also have legitimate host reads. Some
attention families retain diagnostic host fallbacks after graph-construction
failure; the common path does not establish that every recovery path avoids copies.

Sampling keys must remain per request through prefill, acceptance, residual and
bonus draws. Acquire random keys outside compiled graphs to avoid captured-key
reuse. Handle exact-zero uniforms without ties against masked logits; invalid
mass must not put a sentinel into history. Distribution preservation and RNG
ownership matter more than matching an old incidental global draw order.

SSD remains the secondary tier; no permanent RAM duplicate is introduced.
The pinned vLLM version also supports a CPU primary tier plus filesystem,
object-store or peer secondaries, so “vLLM is RAM-only” is not a useful premise.
[vLLM offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/).
oMLX's serial `mx.load` experience and MTPLX's idle-lane snapshot encoding support
keeping GPU state access on its owner while overlapping CPU I/O.
[oMLX preload](https://github.com/jundot/omlx/blob/e467261edc786efd33b1e9023d5c4a827f8aa1c1/omlx/cache/paged_ssd_cache.py#L4217-L4226),
[MTPLX snapshots](https://github.com/youssofal/MTPLX/blob/13297feea79b60b957a6f374f21352087ac45dd1/mtplx/session_bank.py#L2134-L2156).

Code: [paged adapter](../../../crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs),
[pool staging](../../../crates/mlx-paged-attn/src/layer_kv_pool.rs),
[cold cache](../../../crates/mlx-paged-attn/src/cold_cache.rs),
[sampling](../../../crates/mlx-core/src/sampling/dense_draw.rs).

## Measurements and decisions

These are local workload results, not universal speedups. Gemma and native MTP
comparisons used baseline/candidate/candidate/baseline order, one warmup and two
measured rounds per occupancy per arm. Builds, other GPU tests and trace export
were excluded; SSD persistence was disabled for those model comparisons.

| Workload                                             | Baseline → candidate                   | Interpretation                                                           |
| ---------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| Gemma4 12B DSpark, 4 requests × 128 greedy tokens    | 10.569 → 6.111 s, against `bbc3157c`   | 42% lower aggregate latency; 1.73× aggregate throughput                  |
| Qwen3.5 4B NVFP4 MTP, 2 requests × 128 greedy tokens | 1953.9 → 1671.8 ms, against `4bf6b8b1` | 14.4% lower latency; singleton 955.7 → 921.1 ms                          |
| Qwen3.6 35B A3B MXFP8 MTP, same workload             | 2754.8 → 2040.2 ms, against `4bf6b8b1` | 25.9% lower latency; singleton 1412.9 → 1032.3 ms                        |
| Ordinary Qwen, one-token metadata reuse              | Approximately 1–2% lower batch latency | Measured at occupancies 1, 2, 4 and 8                                    |
| SSD upload stage, 14 / 56 MiB                        | 1.666 → 0.454 / 6.467 → 1.364 ms       | Excludes filesystem reads and checksum work                              |
| BF16 prefill head, H=2048, V=65536, 64 / 512 rows    | 903.5 → 700.4 / 2401.5 → 717.6 µs      | Last-row projection only, not total TTFT; 40 alternating measured rounds |
| Speculative residual correction                      | 349–394 → 250–267 µs                   | Removes one completion round trip per rejection                          |

Keep the negative results when choosing future optimizations:

- Direct paged prefill at context 32768 improved width 16 from 5304 to 2191 µs,
  but regressed width 157 from 4943 to 18700 µs. Keep the measured short-suffix
  dispatch boundary rather than switching all prefill to paged attention.
- Fresh-cycle native MTP lost acceptance: dense cycles rose from 41 to 56 and
  N=2 latency from about 2.13 to 2.38 s. Committed draft history replaced it.
- Adaptive Gemma N=4 improved its former exclusive lane from about 10.19 to
  6.45 s, but fixed scheduling was faster at 5.65 s and adaptive N=1 regressed
  about 10%. Calibration must earn its cost.
- Muse scheduling showed inconsistent throughput; multi-token metadata reuse
  regressed both candidate arms and was reverted. Keep scheduling opt-in.
- Ordinary Qwen asynchronous completion showed no reliable gain. Preserve its
  synchronous completion and drain pending forced-token work before error cleanup.
- GPU sampling benefits from grouping and larger vocabularies: a single 32768-way
  draw took 176.8 µs versus 45.6 for a CPU scan; seven grouped GPU draws took
  37.9 µs each. Device residency alone does not establish lower latency.

Packed GEMV/GEMM reductions can change long greedy outputs across batch shapes.
Singleton MTP outputs matched; the second N=2 owner's output differed between
baseline and candidate but repeated within each arm. Gemma teacher-forced checks
retained argmax in 52 inspected rows; changing peer content/order at fixed width
preserved the inspected owner's logits. These support bounded numerical agreement
and owner isolation, not universal bitwise or greedy-token invariance.
[MLX M5 GEMV change](https://github.com/ml-explore/mlx/pull/3888).

## Metal backend boundaries

MLX already owns shared events, dependency tracking, concurrent encoders, residency
sets and gated NAX dispatch. Improve that integration before adding competing
queues or a separate tensor backend. A worker trace contained 21,732 compute
intervals whose union occupied 83.1% of a 4.826 s interval; its gaps included
prefill and turn transitions, so they are not all avoidable CPU encoding time.

Metal 4 command allocation, argument tables and indirect command buffers need a
measured encoding bottleneck plus correct resource/binding lifetimes. Metal IO
needs allocator/storage integration that preserves checksum, identity, immutable
capture bytes and completion-gated publication. Sparse heaps require independent
device support and completion-safe remapping. Pipeline caching is a cold-start
measurement, separate from steady decode. None is an established gain merely
because the API is available.
[Metal 4](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api),
[resource loading](https://developer.apple.com/documentation/metal/resource-loading),
[indirect encoding](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu),
[feature tables](https://developer.apple.com/metal/feature-sets/),
[GPU counters](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers).

## Regression gates and reproduction

Future changes should cover mixed AR/speculative rows; full/partial/zero commit;
unequal prompt positions; both MoE projection modes; cancellation and failed-owner
isolation; allocation/COW/rollback; warm prefixes; and separate-process SSD reuse.
Use the original whole-turn verifier as a numerical oracle, not another call to
the new batch route. Runtime RNG fixtures isolate scheduling with a fixed
non-uniform head; numerical fixtures retain nonconstant weights.

At `b7ff7f23`, 3,422 core tests passed (110 ignored; 18 int8 cases excluded),
and 3,294 TypeScript tests passed (38 skipped). Native build, Clippy, formatting,
typecheck, explicit dense/MoE verifier oracles and transaction fault gates passed.
Run GPU tests serially with `MLX_TEST_REQUIRE_METAL=1`; finite-difference tests
need an explicit `MLX_ENABLE_TF32=0` run. An optional Gemma mobile checkpoint was
absent; ignored/checkpoint-gated cases are not executed coverage. Remote CI status
belongs to the [PR checks](https://github.com/mlx-node/mlx-node/pull/138/checks).

At that revision, real dense/MoE continuations reused 119 and 116 tokens for two
owners and completed sampled MTP smokes. Separate capture/restore processes each
reused 400 tokens per owner and installed both recurrent sidecars, with no queue
drops, write errors or corruptions. These validate cache reuse, not latency or
seeded sampling parity.

The 2026-09-07 admission fixes passed 3,424 core tests with the same exclusions and
both explicit dense/MoE verifier oracles. New regressions cover cached starts and
preemption replay alongside a speculative peer, including token history and
draft-reservation release, plus adaptive admission without a usable decoder.
The checkpoint runs above predate the cached-prefix AR fallback; the updated
continuation and SSD restart scripts require cached turns to reuse target state
without MTP cycles.

```sh
MLX_TEST_REQUIRE_METAL=1 cargo test -p mlx-core --lib -- \
  --nocapture --skip models::qwen3_5::int8_gemm::tests --test-threads=1

MLX_CONTINUOUS_BATCHING=1 MLX_PERSIST_PAGED_CACHE=0 MLX_AGENT_METRICS=0 \
  oxnode docs/research/inference-2026-09-05/benchmarks/mtp-concurrent.ts \
  /absolute/binding/index.cjs /absolute/model /tmp/mtp.json revision-label Qwen35 1,2 2
```

Use `Qwen35Moe` for MoE. [mtp-continuation.ts](benchmarks/mtp-continuation.ts)
checks warm reuse and sampled requests. [mtp-ssd-restart.ts](benchmarks/mtp-ssd-restart.ts)
accepts `binding model output family phase`; set persistence to `1` and use an
isolated `MLX_COLD_CACHE_DIR`, then run `capture` and `restore` in separate processes.
[dspark-concurrent.ts](benchmarks/dspark-concurrent.ts) covers fixed/adaptive Gemma;
the other [benchmark scripts](benchmarks) cover AR, prefixes and host exports.

## Reference revisions

Source behavior was inspected at these revisions; web sources were accessed
2026-09-05–06. Refresh them before using this snapshot to describe upstream today.

| Project                   | Revision                                   |
| ------------------------- | ------------------------------------------ |
| mlx-node initial baseline | `100a03ad10a14a7eab7d816773c072e39dc10206` |
| vLLM                      | `874df9373dab532543a0229fb2f144f7c14093ae` |
| MTPLX                     | `13297feea79b60b957a6f374f21352087ac45dd1` |
| oMLX                      | `e467261edc786efd33b1e9023d5c4a827f8aa1c1` |
| mlx-vlm                   | `d68a25e71e842e8924a54bb3d84d3a3b4d4a2ee1` |
| vendored MLX              | `6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb` |

Additional batching references: [mlx-vlm grouped evaluation](https://github.com/Blaizzy/mlx-vlm/blob/d68a25e71e842e8924a54bb3d84d3a3b4d4a2ee1/mlx_vlm/generate/ar.py#L1206-L1315),
[MTPLX batched decisions](https://github.com/youssofal/MTPLX/blob/13297feea79b60b957a6f374f21352087ac45dd1/mtplx/batched_decode.py#L504-L519).
