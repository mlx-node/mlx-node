# Qwen3.8-Flash-Next: implementation and mlx.fast research

Research: 15–17 September 2026. Starting revision:
`51bb61991479639c5237161652cc22223f3d60ee`. This report consolidates the
initial performance record, mlx.fast investigation, prefill port log, and
macOS 27 recheck. The [runtime guide](../qwen38-flash-next.md) covers usage.

## Reviewer follow-up, 17 September

The read-only GGUF asset-directory finding was reproduced through the packaged
addon: a tiny checkpoint with mode `0444` in a directory with mode `0555` failed
with `Permission denied (os error 13)` before loading tokenizer assets. Qwen4 now
uses the shared writable native GGUF cache, including `MLX_NATIVE_GGUF_CACHE_DIR`,
XDG/OS user-cache selection and temporary-directory fallback. The source identity
and atomic directory publication remain unchanged. The existing cache helpers now
have family-neutral names and serve Qwen4 alongside the other native GGUF loaders.

The regression checks that generated tokenizer/config files are complete and
reusable, the read-only checkpoint stays byte-identical, and no sibling files or
temporary cache directories remain. The source-replacement regression continues
to use the same generated-asset path with an isolated cache directory.

Validation passed: canonical native build and both packaged Metal-library checks;
**3584 core unit tests** (122 ignored; the same three debug-only release
exclusions); strict all-target Clippy; and **197 selected TypeScript tests**. The
shared cache-selection and Qwen3.5 read-only-source regressions also passed.
Generated declaration copies match. Evidence is outside the repository under
`~/Library/Caches/mlx-node/pr154-review-followup-20260917/`.

## Production error-handling audit, 17 September

The production `unwrap`/`expect` audit covers the new Qwen4 runtime,
the extracted Qwen vision code, shared GGUF changes, scheduler/stream integration,
and native build additions. Three subagents independently reviewed runtime state,
weight/cache storage, and shared/native integration. Production assumptions now
use validated pattern matching or contextual errors: missing gates or state,
invalid tensor dimensions, incomplete route/embedding rows, slot mappings,
packed payloads and split-file selection. Optional kernels retain their existing
fallback behavior when quantization companions are absent.

Non-test Clippy guards prohibit `unwrap` and `expect` in the Qwen4, shared Qwen
vision, stream and engine-vision modules, with scoped guards on the shared GGUF
and build helpers. Test assertions remain in test-only code. Native build source
and preamble errors propagate with file context. The focused regressions use
small malformed inputs and fixtures; they do not load the released checkpoint.

A production-only Clippy run, matched against the PR diff from
`ae9ea3b65c92ab7b0b0489704909768e9083ef50`, reported no `unwrap_used` or
`expect_used` diagnostics on added Rust lines. The three source reviews also
covered inactive conditional paths and new build helpers. Existing calls outside
the PR remain outside this audit.

Validation after the audit passed: canonical native build and both packaged
Metal-library smoke checks; **3583 core unit tests** (122 ignored and the same
three debug-assertion-only cases excluded in release); strict all-target Clippy;
workspace TypeScript checking; and **197 tests across eleven selected loader,
discovery, registry, session and Metal-library suites**. All six new malformed-input
and state regressions passed. Generated declaration copies still match. Logs and
the compiler-to-diff audit are retained outside the repository under
`~/Library/Caches/mlx-node/unwrap-audit-20260917/`.

## Whole-branch review, 17 September

Three independent reviews covered the native bridge/shaders, decoder/scheduler,
and storage/media paths. The integration review covered shared vision, stream
residency, TypeScript discovery/loading, documentation and live PR findings.
Caller tracing retained the active fallback implementations, optional kernels,
and independent F32/BF16, paged, MTP and M-RoPE oracle fixtures.

The review fixes three defects:

- Residency-set creation now copies the Metal error description before its
  autorelease pool drains, avoiding a dangling error pointer on failure.
- Qwen4 tokenizer assets use the shared GGUF source identity (path, file size,
  and Unix device/inode/change time). A same-size checkpoint
  replacement preserving modification time therefore generates fresh assets.
- Model discovery ignores the generated asset and temporary publication
  directories, which contain config/tokenizer files but no model weights.

Cleanup removes duplicated hidden-state bookkeeping, unused residency inspection
hooks, a test-only shared-prefill FFI export, unreachable injection shader modes,
obsolete forwarding wrappers, and the one-off inference logit file export.
GDN projections and shared-expert setup now each have one implementation; a
common native BF16 SiLU table replaces two retained copies. Shared image-token
constants and ordinary tracing replace copied literals and a family-specific
wired-limit diagnostic switch. Stale runtime descriptions and a misnamed test
module now match the current execution paths. Existing numerical regression
coverage remains, including shared-prefill parity through production dispatch.

The source-replacement regression publishes and rebuilds real tokenizer assets
using a tiny fixture with no tensor reads. The discovery regression checks both
completed and in-progress asset directories alongside the source GGUF.

Validation passed: canonical native build and both packaged Metal-library smoke
checks; **3577 core unit tests**, with 122 ignored and three existing
debug-assertion-only cases excluded in release mode; strict all-target Clippy;
workspace TypeScript checking; **150 tests across ten loader/discovery/registry
and session suites**, plus **47 Metal-library selection tests**. Batched-prefill
parity also passed with each GDN gate switch enabled separately and both enabled
together. No checkpoint load or throughput comparison was needed for this review cleanup. The residency
failure-path ownership fix was inspected and compiled; a driver allocation
failure was not forced. Logs remain outside the repository under
`~/Library/Caches/mlx-node/branch-review-20260917/`.

## Architecture review, 17 September

The Metal shader audit separates implementation provenance from model-specific
semantics. Thirteen Qwen4-prefixed shader includes are reusable operations and
now live under `crates/mlx-sys/src/metal/common/`, beside the existing recurrence
and quantized kernels. The shared quantized preamble builders and precise
sigmoid helper also live there. Fixed 512-way/top-10 routing, the packed shared
expert, complete GDN geometry and hyper-connection fusions remain in `qwen4/`.
The family adapters keep their existing shape checks, environment controls,
fallbacks and arithmetic. Reuse requirements are recorded in the
[kernel guide](../../crates/mlx-sys/src/metal/README.md).
Two unreferenced tape shader templates were removed; accepted-prefix replay
continues through the existing Rust `GdnKernelTape` and per-step recurrence.

The shader reorganization passed the canonical native build and packaged Metal
library checks, strict all-target Clippy, **3576 core unit tests** (122 ignored;
the same three release-inapplicable assertion tests excluded), and **47 Metal
library selection tests**. A source comparison checked all 45 relocated shader
bodies for unchanged arithmetic and layouts after normalizing renamed helpers,
comments and compile-time assertions. Logs are under
`~/Library/Caches/mlx-node/metal-layout-20260917/`. The full-checkpoint and
performance results below predate this reorganization.

The shared Qwen vision implementation now lives in
`crates/mlx-core/src/vision/qwen/`: processor geometry, encoder, normalized
weight loading, prompt expansion, M-RoPE positions, and the dense/MoE image-feature
cache with its live memory planner. Qwen3.5 dense,
Qwen3.5 MoE, and Qwen4 import that module directly. The previous family-owned
modules and compatibility aliases are removed; numerical and image-layout
regressions move with their implementation. Generic message image extraction
lives in `engine/vision.rs`, also used by Qianfan OCR.

Qwen4 exposes the validated image-capability snapshot and the exact expanded
prompt planner to ChatSession. Planning and prefill share its processor
configuration and input limits (four images, 16 megapixels and 32 MiB per
image), so context budgeting does not use the unexpanded template length.
No vision tensors or inference caches are allocated by the planner.

The existing TypeScript family registry also owns discovery's vision markers
and GGUF policy. A single file/directory path applies that policy, preserves
Qwen3.5's XL restriction and Gemma/Muse's sidecar requirements, prefers converted
target weights, and publishes only Qwen4's first split. Header-only GGUF
checkpoints no longer need an unrelated config file to appear in discovery.

Split filename resolution is a shared GGUF metadata utility and accepts
mixed extension casing across siblings. Missing/ambiguous parts still fail.
Failed exclusive requests release empty owner and page state; a rejected turn
with an existing usable frontier preserves that owner's history.

The Qwen4 decoder, routed-expert residency, hashed PLE and HC/GDN kernel
contracts remain family-owned: other models do not share their tensor layout,
precision boundaries, or SSD ownership policy. Existing generic scheduler,
paged-cache, stream-limit and Metal library-cache modules remain the reuse
boundaries. Experimental kernel switches with independent parity coverage
are retained; the previously rejected MoE replay prototype remains archived
outside the PR. This refactor makes no new performance claim.

Architecture validation:

- The release core suite passed **3576 tests**, with **122 ignored** and three
  existing debug-assertion-only cases excluded in release mode. This includes
  the moved vision/cache regressions, mixed-case split payload reads, exact
  Qwen4 image token planning, repeated sync/stream media failures, and
  cancellation that preserves a usable owner.
- Canonical native build and both packaged Metal library checks passed.
  The addon SHA-256 is
  `5a8f30626b7975d58b293222949970e78d39a7d1e04521045e2a53b51e00df0f`;
  both Metal libraries match the preceding attention-gate build.
- Strict all-target Clippy passed for `mlx-core` and `mlx-sys`. Workspace
  TypeScript checking and formatting/type-aware lint on the four changed
  TypeScript sources passed. Both generated declaration copies match.
- The six affected TypeScript suites passed **95 tests**, including the
  Qwen4 discovery cases in the existing agent suite. No new standalone
  TypeScript test harness is committed.
- The exact-arithmetic full-checkpoint smoke passed **11 requests** with the
  matching auxiliary checkpoint, plus public image capability/count checks
  and **eight rejected malformed-image owners** followed by successful red,
  continued-red and changed-blue image turns. The 64-by-64 image planned 64
  image tokens; a five-image conversation was rejected before inference.
- Live headroom admitted **60 GiB / 345 slots** under the unchanged 63 GiB
  cap. Peak physical footprint was **61,956,784,896 bytes**; all **1365 pages**
  were free after reset and reserved state was zero. The memory guard passed.
  Compilation overlapped this correctness-only smoke; its timings are not
  performance results.
- Raw evidence and immutable native artifacts remain outside the repository
  under `~/Library/Caches/mlx-node/qwen4-reference-gap-20260917/architecture-*`.

## 95% follow-up: refreshed reference

The 17 September follow-up inspected the new promoted tree
[`05944553`](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/tree/05944553c77666f74651f5c20d6cb4b8b132b7d9),
including [PR #763](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/pull/763)
and [PR #828](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/pull/828).
The latter's official run, `35120094932`, reports candidate-leg times of
0.000398337728515625 seconds per prefill token and 0.01558816650390625 seconds
per decode token. Their reciprocals are **2510.433 / 64.151 tokens/s**, making
this pass's 95% targets **2384.911 / 60.944 tokens/s**. The live leaderboard
renders a different decode figure; this comparison uses the explicit raw
seconds-per-token fields. The artifact still reports serial mode, depth zero,
and 64 checked steps. The older reference and results below remain historical.

The source delta contains three mechanisms:

- **Ordered split-K mixer:** four SIMD groups calculate separate input blocks,
  retain each block's result, and restore the original ascending fold before
  the SIMD reduction. This addresses the long, narrow mixer down/injection
  projections. The local port retains affine8 packs, FP16 companions, BF16
  projection/division/activation boundaries, and the existing weight banks.
- **PLE host context mirror:** our decoder already builds n-gram IDs from its
  owned host token history, so it does not have the GPU history readback this
  reference change removes.
- **Compiled MLP replay:** the reference keeps this disabled by default because
  its local gain did not improve the ranked result. Capturing complete expert
  banks is also inappropriate for our mutable bounded slots; any local graph
  replay must retain slot arguments, reader ownership, and route validation.

A fresh unchanged `2b66b088` baseline (`target95-baseline`) passed the existing
63 GiB/393-slot guard and all five standard output hashes. Its three measured
prefill rates were 2318.489/2304.872/2336.590, and decode rates were
54.955/55.209/55.204 tokens/s. This slower baseline is retained; a new candidate
must be compared with contemporaneous controls. A separate 256-output CPU
sample showed substantial waiting for expert route readbacks, but that longer
continuation is not pooled with the standard 128-output timing comparison.

The first ordered split-K native check matched independent projections and the
105-test Qwen suite passed. A ten-request control/split comparison passed all
hashes and guards, but its GPU temperature rose from 56.3 C to 98.1 C and both
variants slowed. Control medians were 1942.223/50.841 and split medians were
1757.375/53.114 tokens/s. A separate per-request 60 C gate also passed hashes
and guards, but produced low, variable clock/rate observations: control medians
730.645/38.267 and split medians 814.072/48.165. Neither run establishes a stable
speedup or the 95% target. Both are retained in `target95-split-comparisons.json`.
The user subsequently agreed to leave browser/video activity idle for new
comparisons; that does not retroactively isolate these runs.

The next candidate also ports `TrackFastGDNDecode`'s initial state-row prefetch,
register-held q/k vectors, and vector state loads/stores. Each value row retains
its original ascending FP32 products, SIMD sum, modulo head mapping and BF16
output boundary; recurrent state and convolution history remain separate owned
outputs. Earlier standalone probes are recorded above/below as diagnostics, not
a demonstrated full-model gain. `MLX_QWEN4_GDN_DECODE_PREFETCH=1` selects it only
inside the already eligible M5 singleton fused path.

After adding the GDN port and checking the split-mixer switch through the
weight wrapper, all **106 Qwen native tests passed** (43.77 seconds). The GDN
regression interleaves two owners and changing inputs/weights across 24 steps
for convolution kernels of lengths 2, 4 and 8, comparing all output, recurrent
state and history values against the independent operation sequence.

### Fresh-process verification and remaining dispatch work

The `target95-next-build` addon (SHA-256
`2077d758e1ea45ad6413f6f8eaed04ca54830f18fb8a157673212ca951a8e5b2`)
passed its canonical build. Four fresh processes ran control/candidate/candidate/control,
each with two warmups and three measured requests. Both variants used the same
binary, 1024/128 token counts and 63 GiB/393-slot plan. All 20 output hashes and
all four memory/pressure guards passed.

| Process                      | Prefill median | Decode median |
| ---------------------------- | -------------: | ------------: |
| Control A                    |       2321.043 |        55.576 |
| Split mixer + GDN prefetch A |       2256.194 |        56.129 |
| Split mixer + GDN prefetch B |        785.992 |        40.699 |
| Control B                    |        828.839 |        43.390 |

These runs do **not** demonstrate a repeatable gain or 95% performance. Median
GPU frequency among samples with over 90% GPU activity changed from 1618/1609 MHz
in the first pair to 844/830 MHz in the second. Maximum reported GPU temperature
was 85.4/91.8/73.4/62.2 C respectively. AC power and mode 2 remained selected;
macOS reported no recorded thermal/performance warning, and swap usage did not
increase. These observations do not establish why GPU clocks fell. Full results
are retained in `target95-next-abba.json`; slower samples are not discarded.

A separate diagnostic, `target95-next-profile128`, used the actual 128-token
continuation and traced route commits while sampling the host stack. Each of
requests 1–4 recorded 127 hits and zero replays. Only cold request 0 missed,
at positions 1024/1041/1058/1075/1092/1109/1126/1143. Thus the earlier 256-token
readback profile does not explain this warmed workload. Of 3182 model-thread
samples, 1276 were under asynchronous submission, including 782 waiting for MLX
scheduler completion. This profile ran at the lower observed clocks and its
timings are excluded from ordinary comparisons.

The next source ports retain the existing weight storage and memory admission:

- `MLX_QWEN4_ROUTE_SHARED_GATE=1` combines singleton route selection with the
  shared gate, following the reference's combined-dispatch pattern. The reference
  combines a quantized gate; our BF16 gate instead keeps MLX's existing small-N
  GEMV lane walk, reduction and BF16 output. Probability-first route rounding,
  stable ties and normalization remain unchanged. Existing resident BF16 gate
  weights are graph arguments; unsupported shapes/dtypes retain the original path.
- `MLX_QWEN4_MIXER_UP_COLUMNS=1` uses `TrackFastMixer.upMixSource`'s two-column
  group and eight independent stream products. It reads the original row layout,
  adds no packed-bank copy, and preserves affine8 products, BF16 sigmoid/mean and
  injection boundaries. It applies to the existing eligible fused injection path.

Both remain opt-in. All **108 Qwen native tests passed** (38.64 seconds),
including the combined route/gate test under all four cached-graph/register
settings with 64 changing, strided input/weight cases each. The mixer tests
compare independent projections and native sigmoid halfway cases. Full-model
validation follows separately. These kernel patterns are separate from the
reference's disabled-by-default full MLP graph capture.

The final four-port addon (`target95-dispatch-build`, SHA-256
`7f75c23f2e7657b23bbf4fe1ff21416828f2f007f5a600842fb0370fc8d86475`)
passed the canonical package build and strict all-target Clippy for `mlx-core`
and `mlx-sys`. Its exact-arithmetic complete-checkpoint smoke passed all **11
requests**: AR, streaming continuation, native/adaptive MTP, concurrent owner
continuation, cancellation recovery and image replacement. All 1365 pages and
state reservations were released. The auxiliary-model plan remained 63 GiB /
366 slots, peak physical footprint 65,203,291,888 bytes. Memory and pressure
guards passed; the smoke allowed concurrent compilation and supplies no
throughput evidence. BF16 prefill/PLE were disabled for this exactness smoke.

The next candidate/control/control/candidate sequence (`target95-dispatch-abba`)
again used four fresh processes, two warmups and three measurements each,
1024 input / 128 output tokens, and 63 GiB / 393 slots. Every hash and guard
passed. No compilation or native GPU tests overlapped these measurements.

| Process               | Prefill median | Decode median |
| --------------------- | -------------: | ------------: |
| Four-port candidate A |       2331.351 |        56.323 |
| Control A             |       2259.056 |        54.002 |
| Control B             |       2257.018 |        54.347 |
| Four-port candidate B |       2243.066 |        54.255 |

The six-sample control medians were 2258.037/54.032; candidate medians were
2294.530/55.364 tokens/s. Candidate ranges were 2224.398–2336.896 prefill and
54.188–56.779 decode. The candidate's first-process advantage did not repeat
against the second control, so this is not grounds for default promotion.
Against the refreshed raw reference, candidate medians are **91.4% prefill /
86.3% decode**; **the 95% objective is not achieved**. These measurements use
the optional BF16 prefill/PLE configuration, with its existing numerical limits.
The earlier 90% result remains a separately dated result, not a substitute for
these contemporaneous measurements.

### Cached MoE replay experiment (archived)

An external prototype, `MLX_QWEN4_MOE_REPLAY=1`, tested a narrower adaptation of
`TrackFastMLPReplay`'s cached-graph pattern. Router projection, probability-first
route/shared-gate selection, logical-to-slot lookup and routed/shared expert
kernels become one retained MLX graph. Each call supplies the current input,
router/shared-gate weights, slot map, six weight banks and native sigmoid table.
It captures no mutable weight bank or route map as a constant. Existing kernel
arithmetic is unchanged; this experiment targets host graph construction.

The prototype required the matching router-row, route/shared-gate, register,
expert-lane and two-row switches, and stays inside tentative singleton routing.
Its selected IDs join the existing route tape; the output joins existing slot
readers. The same commit, miss rollback/replay and cancellation handling remain
in force. No bank capacity, system reserve or completion guard is increased or
removed. The ordinary path remains available for unsupported inputs/settings.

In the prototype tree, all **109 Qwen native tests passed** (47.02 seconds). The added replay regression
compares independent BF16 router/shared-gate projections and the already checked
separate expert operations across Q4_K/Q5_K gate/up and affine5/8 down banks,
changing inputs/weights, replacement slot maps, and missing slots. IDs and every
output match exactly; unsupported multi-token input falls back. The canonical
addon build and strict all-target Clippy also passed. The immutable addon is
`target95-replay-build`, SHA-256
`3c9c2eeb7a8071d656d3e2036671c97dbfc1746963b356df85d9e710abd3f03b`.
A full-checkpoint diagnostic forced rollback for every tentative token across
two 1024/128 requests: **254 replays**, comprising 126 actual misses and 128
hits, all with the expected standard output hash. The trace explicitly confirmed
the compiled MoE path executed. The 63 GiB/393-slot memory/pressure guard passed
at 69,366,627,080 bytes peak physical footprint. These forced-replay timings are
excluded from performance comparisons.
The replay-enabled complete-checkpoint smoke also passed all 11 requests and
released every page/state reservation, with an actual auxiliary-model plan of
63 GiB / 366 slots and 65,159,760,056 bytes peak physical footprint. Its trace
confirmed graph execution. These are correctness checks, not throughput samples.

The first fresh pair before the user closed applications gave prototype/control
medians of 1499.895/51.441 and 1473.065/51.033 tokens/s. These rates and all
output hashes/guards are retained as `target95-replay-candidate-a` and
`target95-replay-control-a`; they are not pooled with the next series.

After the user offered a five-minute closed-application window, a separate
control/prototype/prototype/control series used the same immutable replay addon,
1024/128 tokens, two warmups plus three measurements per process and 63 GiB /
393 slots. Every output hash and all four comparison guards passed.

| Process      | Prefill median | Decode median | Busy GPU MHz median |
| ------------ | -------------: | ------------: | ------------------: |
| Replay off A |       2278.178 |        54.467 |                1618 |
| Replay on A  |       2261.055 |        54.807 |                1618 |
| Replay on B  |       2237.141 |        54.516 |                1612 |
| Replay off B |        999.411 |        47.458 |                1125 |

The requested window began at 03:23:39 UTC. The first pair ended at 03:27:57;
the second prototype ended at 03:28:45 and the final control at 03:29:24, beyond
that five-minute window. The full sequence cannot be described as a controlled
closed-application comparison. The first pair supplies no convincing replay
benefit, and pooling the slow final control would exaggerate it. GPU-clock
variation remains an observation with no established cause. The prototype's
six-sample median was 2252.210/54.693 tokens/s, below the 95% targets; its ranges
were 2156.429–2280.691 and 54.439–55.049. Full data and telemetry are retained in
`target95-closed-abba.json`.

The replay source patch, regression and immutable addon are archived outside
the PR (`target95-replay-final.patch`, `target95-replay-build`). With no
repeatable benefit, the replay runtime branch and its flag were removed from
the shipping source. The forced-rollback and smoke results above describe that
archived prototype, not a newer runtime configuration.

### Attention output gate follow-up

The reference's
[`attnGateSource`](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/blob/05944553c77666f74651f5c20d6cb4b8b132b7d9/Runner/FastModel/TrackFastKernels2.swift#L455)
reads the attention and projected gate directly in one operation. Our Q/K
normalization already handles strides without materializing the input views;
there is no missing copy removal there. The output path still reshaped the
head-major attention and interleaved gate into token-major buffers before its
pointwise multiply. `MLX_QWEN4_ATTENTION_GATE=1` ports direct addressing for
that output, preserving the local interleaved head layout and using the existing
native BF16 sigmoid table. All changing arrays and strides remain graph inputs;
unsupported shapes/dtypes and disabled fused pointwise operations retain the
original path. No additional weight copies or admission changes are introduced.

All **109 Qwen native tests passed** (53.05 seconds). The new gate regression
compares the independent native compiled sigmoid/multiply across changing
inputs, offset/transposed views, singleton and 7/64/1024-token windows, and
1/8/24 heads. It covers every finite BF16 gate value, signed zeros, infinities
and native exponential halfway cases, with bitwise output comparison. F32
inputs are rejected by the custom path. An initial broader Cargo invocation
also began building unrelated integration binaries; those task-owned compiler
jobs were stopped and the completed Qwen unit-test binary was run directly.
The stopped compilation supplies no integration-test result.

The canonical addon build, packaged metallib checks and strict all-target Clippy
passed. `target95-gate-build` identifies the immutable addon, SHA-256
`f2659ffa435bde3ec767d242f00498a2d02e45a48c631f502da658786b3f77bc`.
Its gate-enabled exact-arithmetic complete-checkpoint smoke passed all **11
requests**, including AR, streaming continuation, native/adaptive MTP,
concurrent owners, cancellation recovery and image continuation/replacement.
All 1365 pages were free afterward and all state/block reservations were zero.
With live headroom, the auxiliary-model plan admitted **62 GiB / 359 slots**
under the unchanged 63 GiB cap; peak physical footprint was 64,151,749,760 bytes.
Memory/pressure guards passed. This is correctness evidence only, not a
fixed-63-GiB performance comparison.

The first alternating gate-off/on comparison (`target95-gate-paired-a`) stopped
before loading: its comparison guard detected Metal compilation in the
`lfm2-vllm-align` / `k2-horizon` worktrees. The guard was not relaxed and those
processes were not stopped. It supplies no throughput sample for this port.
The gate remains opt-in; the 95% objective is still unachieved.

## Previous 90% result and limits

**The previous optional configuration exceeded 90% of both then-published rates.**
Six measured requests across two fresh processes used the supplied checkpoint,
1024 input tokens, 128 generated tokens, two warmups per process, and the same
63 GiB/393-slot memory plan. Every measured request exceeded both thresholds
and preserved the standard output hash; both process guards passed.

| Measurement                         |  Prefill tokens/s | Decode tokens/s |
| ----------------------------------- | ----------------: | --------------: |
| Published reference                 |           2512.69 |           63.26 |
| 90% threshold                       |          2261.421 |          56.934 |
| Latest candidate, six-sample median |      **2412.472** |      **58.078** |
| Percentage of published reference   |         **96.0%** |       **91.8%** |
| Six-sample range                    | 2398.751–2416.677 |   57.915–58.248 |

The two process medians were 2411.633/58.111 and 2413.311/57.939. This is an
ordinary local workload comparison with the published rates, not a matched
engine comparison: the reference uses a different checkpoint and private
prompt. Earlier configurations and repeated controls sometimes slowed sharply
with falling GPU clocks; those results remain below. The latest result does
not establish performance under every thermal or desktop-load condition.

The measured configuration uses **optional BF16 prefill operands and wide PLE
GEMM**, which can change arithmetic and continuation. They remain off by
default. Native exactness tests for the new load/routing ports do not erase
those separately documented numerical differences.

### Measured configuration

Use these settings before starting the process. Other Qwen4 controls retain
their documented defaults; the measured mixer down/injection fusion is disabled.
The supplied GGUF codes, system reserves and admission rules are unchanged.

```sh
export MLX_METAL_HASH_KERNEL_CACHE=1
export MLX_QWEN4_ATTENTION_NORM_ROTARY=1
export MLX_QWEN4_DECODE_ASYNC_PLE=1
export MLX_QWEN4_EXPERT_LANE_STAGING=1
export MLX_QWEN4_FUSED_ROTARY=1
export MLX_QWEN4_GDN_GATE_INPUTS=1
export MLX_QWEN4_GDN_GATE_PAIR=1
export MLX_QWEN4_GDN_VECTOR_ROWS=1
export MLX_QWEN4_INJECT_NEXT_NORM=1
export MLX_QWEN4_MIXER_DOWN_INJECT=0
export MLX_QWEN4_MIXER_INJECT=1
export MLX_QWEN4_MIXER_LANE_PRODUCTS=1
export MLX_QWEN4_PREFILL_PLE_GEMM=1
export MLX_QWEN4_PREFILL_REFERENCE_BF16=1
export MLX_QWEN4_PREFILL_SHARED_COMBINE=1
export MLX_QWEN4_REFERENCE_DOWN_SCHEDULE=1
export MLX_QWEN4_ROTARY_TABLES=1
export MLX_QWEN4_ROUTER_ROW_SCHEDULE=1
export MLX_QWEN4_ROUTE_REGISTER_RESULTS=1
export MLX_QWEN4_WEIGHT_CACHE_GIB=63
```

Evidence: `register-reference-a` and `register-reference-b` under
`~/Library/Caches/mlx-node/qwen4-reference-gap-20260917/`. The immutable addon
SHA-256 is `c08e3947deecb7171db7122cac98ca88f46b764c737760f703da7ff598c50f0e`.
Invocation records contain the harness hash, environment, model path, output
hashes, timing windows, power snapshots and memory-guard results.

## Reference, checkpoint, and method

Primary reference: [mlx.fast source at 8981cef](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/tree/8981cef5a0a0c5b327f72cf040aa566fc61ff723),
[winning PR #580](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/pull/580),
and the [published leaderboard](https://www.yukon.org/mlxfast), as inspected
during this research. The local clone is pinned to
`8981cef5a0a0c5b327f72cf040aa566fc61ff723`.

The reference uses affine-4/group-32 weights, a private 1024-token `botany`
tape, and serial autoregressive generation with depth zero. MTP does not
explain its score. Its winning artifact reports 64 generated steps while the
contract describes 128 checked steps; that discrepancy remains unresolved.
Our supplied checkpoint is `Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL`, with mixed
Q4_K/Q5_K and affine5/8 runtime banks. Both reported machines have a 40-core
M5 Max GPU and 128 GiB RAM. Prompt, quantization, thermals, and desktop load
are not matched.

Ordinary local comparisons use:

- 1024 input tokens and greedy 128-token output, no MTP or prefix reuse, unless
  explicitly labelled prefill-only (one output token).
- A fixed 63 GiB weight budget (67,645,734,912 bytes), 393 partial expert slots
  per layer, and no fully resident expert layers. Cache contents persist
  between legs. Earlier automatic-budget runs are identified separately.
- Two warmups per variant, generally three measured samples per variant, with
  interleaved ABBAAB order. Four-process ABBA rechecks use two warmups and three
  measurements in each process. Counts that differ are stated below.
- Identical input IDs, generated counts and output hashes within each exact
  comparison. Feature environments and immutable addon hashes are recorded.
- A process-group memory/pressure/timeout guard, normally an 88 GiB process
  ceiling and an 8 GiB competing-model threshold. Failed and interrupted runs
  are retained. No compiler or GPU probe overlaps ordinary performance runs.

The guard is not complete desktop isolation. OS file cache and thermal history
remain uncontrolled. A local 60 C request gate is distinct from the reference's
40 C per-phase gate; neither its results nor diagnostic samples are pooled with
ordinary warm runs. The 40 C attempt timed out without producing a sample.

The standard prompt consists of repetitive notes followed by an integer-list
request. A separate synthetic Rust cache-review prompt has 1023 input tokens;
it is not a captured multi-turn coding-agent workload.

| Identity                            | SHA-256                                                            |
| ----------------------------------- | ------------------------------------------------------------------ |
| Standard input IDs                  | `fb82a894b07a6c8c60cba0106e0831b98ed6f581cd525b0eff69df03ec264167` |
| Standard 128-token output           | `e9987578c1b7724ed4a9cf49f355c09600e9e4c04719154da85dd82d54f088a6` |
| Coding output at 63 GiB / 393 slots | `fa17b454f16e746f555cc09a7b4aa93313993ed2e2ce94f8bae7f91cebe45a96` |
| Prefill-only output                 | `6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b` |

## Source findings and shipped implementation

The reference was inspected locally before porting. MIT attribution remains in
[`MLXFAST-LICENSE.txt`](../../crates/mlx-sys/src/metal/MLXFAST-LICENSE.txt).
The ports adapt its execution to the existing GGUF arithmetic and bounded
partial residency; they do not replace the checkpoint or relax its admission.

| Reference source under `Runner/FastModel/`            | Local implementation and adaptation                                                                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `TrackPrefillSort.swift`                              | Ballot counts, stable ordering and inverse maps; partial-bank and tail handling retained.                                                       |
| `TrackFastMoE` and indirect dispatch                  | Device route maps and route tapes, validated before committing an output; missing slots restore and replay the original frontier.               |
| `TrackPrefillMixerAct.swift`                          | 32-row mixer activation and hyper-connection fusions with local BF16 product boundaries.                                                        |
| `TrackFastKernels2.swift`                             | GDN preparation/norm, wide normalization, injection and optional attention normalization/rotary; local reduction and rounding order retained.   |
| `TrackPrefillIndirectMetal.swift`                     | Packed group prefetch and vector stores for Q4_K/Q5_K gate/up and affine5/8 down; default F32/TF32 operands retained where required.            |
| `TrackFastModel.swift` / `TrackP12Prefill.swift`      | Shared Q8 projection pairing, split projections, bounded asynchronous submissions, and final head completion inside the route commit.           |
| `TrackFastPLE.swift`                                  | Direct four-tap convolution; wide PLE GEMM stays opt-in because its arithmetic can change continuation.                                         |
| `TrackBF16Functions.swift`                            | BF16 sigmoid table generated with this checkout's compiled sigmoid-multiply, rather than a different exponential approximation.                 |
| `TrackFastMixer.header1` / `TrackFastMoE.helpersCore` | Smaller custom-kernel helper headers, preserving register/dequantization code. Dense decode source shrinks from about 147 KB to 23,417 bytes.   |
| Reference static feature settings                     | A shared Rust/C++ thread-local settings cache scoped to one forward, refreshed on the next forward.                                             |
| `TrackFastGDNDecode` / `TrackMultiProj`               | Optional direct BF16 gate inputs and singleton pairing of the two dense 48-by-2560 gate projections. Wider windows retain separate projections. |

### Routing, lifetime, and completion

Warm partial banks previously required copying ten expert IDs to the CPU at
every layer. Tentative device routing now records the actual selections across
a token/window, retains every bank reader and owned recurrent/PLE state, and
joins them with the final output. It validates all original IDs against the
unchanged host mappings before publication. A missing slot uses an initialized,
bounds-checked placeholder, then discards that output, restores the recurrent
and paged frontier, and replays through the ordinary loader. A 16-token decode
cooldown bounds repeated misses.

Wide routing uses previous observed experts only as an eligibility hint. Fresh
routes still require commit validation. Slot replacement waits for prior
readers; only one deferred wide expert reduction can survive across a store.
MTP verification retains its separate completion/commit boundaries.

The reference's first-two/then-three-layer asynchronous schedule is bounded
inside these transactions. Final logits, route tapes, PLE/GDN state and page
writes complete together before validation. Optional deferred singleton PLE
was repaired in v25: the one-token branch now forwards the completion flag
while retaining an independent history copy and identical arithmetic.

Compact Q8 singleton projections read BF16 inputs and FP16 companions directly,
preserving lane ownership, K traversal, accumulation and output rounding.
Shared-expert fusions preserve the individual routed/shared products and their
final addition. Compiled graph wrappers receive every changing array explicitly.

### Defaults and optional controls

Eligible exact ports through reference-v9 are enabled, together with smaller
headers, `MLX_QWEN4_CACHE_FLAGS`, `MLX_QWEN4_DEFER_FINAL_COMPLETION`, and split
Metal residency sets. Runtime switches accept `0` for their control path.
Shape, dtype, hardware and transaction eligibility still select fallbacks.

Default prefill controls include `MLX_QWEN4_PREFILL_MIXER_BM32`,
`MLX_QWEN4_PREFILL_HC_MIX`, `MLX_QWEN4_PREFILL_GDN_NORM`,
`MLX_QWEN4_PREFILL_Q8_PREFETCH`, `MLX_QWEN4_PREFILL_SHARED_PAIR`,
`MLX_QWEN4_PREFILL_ROUTER`, `MLX_QWEN4_PREFILL_PLE_CONV`,
`MLX_QWEN4_PREFILL_PACKED_EXPERTS`, `MLX_QWEN4_PREFILL_REFERENCE_INDIRECT`,
`MLX_QWEN4_PREFILL_DEVICE_ROUTES`, `MLX_QWEN4_PREFILL_NORM`,
`MLX_QWEN4_PREFILL_ASYNC_WINDOW`, and `MLX_QWEN4_PREFILL_SHARED_Q8`.

`MLX_QWEN4_PREFILL_REFERENCE_BF16=1` and `MLX_QWEN4_PREFILL_PLE_GEMM=1`
are **arithmetic experiments, off by default**. Historical results exceeding
90% prefill used both. Rotary reuse/fusion, next-layer injection/norm, deferred
PLE (`MLX_QWEN4_DECODE_ASYNC_PLE`), mixer injection, shared combine, reference
down scheduling, attention normalization/rotary and direct/paired GDN gates
remain optional. Their existence is not evidence of a repeatable speedup.

### Bounded residency on macOS 27

The unchanged v19 addon repeatedly hit GPU timeouts on macOS 27. The reference
splits wired allocations across residency sets capped at 5% of the device's
recommended working set, with a 64 MiB floor and at most 32 sets. Porting this
behavior fixed the standard workload; selecting one set on the same new addon
still timed out. `MLX_RESIDENCY_SET_MAX_PCT=0` is the one-set rollback.

The overlay in `crates/mlx-sys/metal-residency` preserves allocation accounting,
locking, oversized-allocation handling, and attachment of new sets to every
queue before command-buffer commit. `build.rs` creates it under `OUT_DIR`, and
CMake plus the bridge consume identical headers. The MLX submodule is unchanged
by this optimization update. CPU/CUDA builds do not use the overlay. The build
also links the compiler runtime needed for the platform availability predicate.

Eleven standalone lifecycle/queue checks passed with at most 256 MiB of test
allocations. That research harness is archived outside the PR. The port does
not raise total wired bytes, weight capacity, expert slots, or system reserves.

Full hot weights occupy 83,783,503,360 bytes; fixed dense banks occupy
5,245,160,960 bytes. Full hot residency needs roughly 100 GiB of the planner's
live available estimate, including reserves/staging/workspace. Recorded
estimates did not support it, and reserves were not lowered. Routed storage is
only about 4% larger than equivalent reference affine4 banks; fixed Q8 storage
is about 1.8 times affine4/group32. Size alone does not explain the prefill gap.

## Measurements

Values below are medians in tokens/s. Comparisons are valid within their stated
run; cross-run differences are not attributed to code. All successful exact
legs preserve the hashes for their configuration.

### Initial support and decode pass

The initial support record measured 1320.2 prefill / 21.39 decode with automatic
72 GiB admission and 454 slots, versus 1292.8 prefill for its local control.
Peak guarded footprint was 73.44 GiB. This predates the reference ports.

| Same-run comparison                              | Control prefill/decode | Candidate prefill/decode |
| ------------------------------------------------ | ---------------------: | -----------------------: |
| Compact Q8, ballot counts, device routes         |        1359.28 / 21.66 |          1409.35 / 31.02 |
| Add GDN preparation and mixer fusion             |        1028.27 / 20.58 |          1156.92 / 31.51 |
| Singleton routing and reader completion          |        1394.19 / 22.16 |          1502.03 / 40.96 |
| Cached graphs and shared-expert fusion           |        1375.76 / 22.14 |          1473.02 / 44.87 |
| Corrected initial defaults, fixed 63 GiB         |        1369.31 / 21.98 |          1476.00 / 44.81 |
| Corrected defaults, automatic 71 GiB / 448 slots |        1351.77 / 21.46 |          1477.57 / 44.86 |
| Coding prompt, automatic 72 GiB / 454 slots      |         541.27 / 17.89 |           587.51 / 31.35 |

The corrected fixed-budget candidate's three prefill samples were
1481.70/1476.00/1467.50; decode 44.90/44.58/44.81. Peak footprint was 64.26 GiB.
The automatic-budget coding run ranged from 549.77–667.72 prefill and
30.68–31.99 decode, with a 73.06 GiB peak. Its output hash was
`241721e02a414d72215aafc838611b604466f4ccf74ccfbef6d2eada081d7417`.
It establishes parity within that capacity, not across cache capacities.

Earlier mixer-fusion results used an exponential approximation later found to
change coding output. A twelve-variant diagnostic isolated the issue. At a
sigmoid input of -6.84375, a BF16 boundary produced -0.032958984375 instead of
-0.03271484375. Generating the 128 KiB lookup table with the native compiled
sigmoid-multiply and a dynamic multiplier of one fixes that regression. The
four-row native regression and full coding output then match. Earlier timings
remain historical experiments, not final validated-default results.

### Prefill ports on macOS 26.6.2 (25G83)

| Run                                     |  Control | Candidate | Scope                                        |
| --------------------------------------- | -------: | --------: | -------------------------------------------- |
| v2 mixer/GDN                            | 1398.748 |  1392.847 | Prefill-only, six samples each; inconclusive |
| v3 Q8/pairing combined                  | 1321.657 |  1367.894 | Prefill-only, three samples each             |
| v4 routing/convolution                  | 1111.079 |  1157.181 | Prefill-only; control had drifted            |
| v5 packed always-indirect               | 1351.973 |  1512.304 | Prefill-only, four samples each              |
| v6 add device routing                   | 1414.352 |  1617.545 | Prefill-only, four samples each              |
| v7 exact ports plus asynchronous window | 1534.903 |  1972.589 | Prefill-only, three samples each             |
| v12 exact versus experimental BF16/PLE  | 2033.948 |  2325.742 | Prefill-only                                 |

V7's candidate samples were 2030.362/1972.589/1854.302; TTFT median fell from
667.143 to 519.115 ms. Isolated packed-kernel loop gains are not substituted
for these end-to-end measurements. Runs that changed arithmetic in one loaded
process sometimes nearly doubled latency; their mixed medians do not isolate
BF16 performance, so later checks used fixed process configurations.

V14's four-process ABBA comparison produced six candidate prefill samples:
2318.837, 2312.574, 2326.658, 2320.915, 2315.283, 2297.203. All exceed the
2261.421 threshold; median 2317.060 is 92.2% of the published reference.
Decode median 52.934 is below its target. V16 measured 2299.093/52.220 with
final completion off and 2302.827/53.989 with it on. Its candidate prefill
samples were 2302.827/2305.139/2292.461, all above threshold.

Later optional-port comparisons retain their slower results:

| Run/configuration                  |  Prefill | Decode |
| ---------------------------------- | -------: | -----: |
| v16 BF16 defaults                  | 2272.352 | 54.179 |
| v16 plus four reuse ports          | 2225.289 | 54.782 |
| v16 plus all earlier exact fusions | 2254.462 | 54.821 |
| v17 control                        | 1987.136 | 53.953 |
| v17 reference down schedule        | 2026.381 | 53.468 |
| v17 down plus decode ports         | 2064.572 | 55.338 |
| v18 attention plus decode ports    | 2089.648 | 55.824 |
| v19 attention plus decode ports    | 1688.352 | 54.002 |

The best listed decode median, 55.824, is about 88.2% of the reference. Late
samples slowed across configurations. No later option was promoted from these
results. The v19 clock decline is observed, but its cause is unproven.

### macOS 27.0 (26A428), High Power recheck

System Settings visibly confirms High Power on the adapter; `pmset` reports
AC `powermode=2`. Simultaneous `system_profiler` labels report Low Power Yes /
High Power No, so those labels cannot establish the selected mode. The earlier
Low Power explanation is withdrawn. No power setting was changed by the agent.

The 20-leg v23 comparison produced:

| Configuration              |  Prefill | Decode |
| -------------------------- | -------: | -----: |
| Attention/decode control   | 1838.928 | 46.137 |
| Direct GDN gate inputs     | 1795.497 | 46.924 |
| Add singleton gate pairing | 1794.451 | 46.417 |
| Disable attention fusion   | 1782.628 | 48.249 |

GPU temperature was 87–94 C and clocks varied. All outputs and memory guards
passed. This does not establish a gain from the optional GDN changes.

A separate residency 5%/2%/2%/5% ABBA comparison used a local 60 C per-request
gate. Process medians were 914.373/49.371, 815.719/48.730, 811.285/49.084 and
761.662/49.184. These are not pooled with warm comparisons. The 5% default stays.

V24 tested reference Metal 4.1 compatibility in four fresh processes, off/on/on/off:
2088.782/49.684, 2087.869/49.637, 2094.635/49.959 and 2104.472/50.116.
Pooled medians were 2092.288/49.875 off versus 2093.466/49.676 on. Sixteen
isolated cross-buffer cases matched, but no model gain was established.
**The compatibility experiment was removed during PR cleanup.**

V25 repairs deferred singleton PLE completion. Its 15-leg comparison measured
1020.780/45.011 with the flag off, 1028.449/45.066 on, and 974.036/45.030 for
a minimal BF16 configuration plus deferred PLE. A fresh fixed-configuration
v24/v25/v25/v24 comparison produced:

| Process |  Prefill | Decode |
| ------- | -------: | -----: |
| v24 A   | 2136.996 | 51.462 |
| v25 B   | 2024.711 | 49.856 |
| v25 C   | 1159.293 | 48.528 |
| v24 D   | 1309.421 | 48.248 |

The unchanged control slows too. Whole-request GPU samples ranged from
1606–1619 MHz in A to 1008–1447 MHz in D, with nearly continuous activity.
One-second samples do not isolate prefill or establish why clocks changed.
All hashes/guards passed; deferred PLE remains off by default.

The subsequent ordinary prefill-only samples were 1343.232/1502.858/1576.088.
A later route-trace diagnostic measured 2117.048/2117.173/2121.934 with all
four warm prefills committing resident hits and no replay. It rules out replay
in that diagnostic, not in every earlier slow run.

## Correctness, diagnostics, and rejected experiments

Default ports retain local arithmetic. Regressions cover mixed quantization,
partial tiles, offset views, changing graph inputs, router ties/NaNs/infinities,
reader lifetimes, paged/recurrent rollback, retained prefixes, MTP frontiers,
multiple owners, cancellation and media positions.

The experimental BF16/PLE candidate is not bit-identical. An eight-case,
128-step teacher-forced replay has 13/1024 strict argmax differences, at most
4/128 for one prompt, with finite logits. It passes the local 10% per-stream
rule, which is neither a general quality evaluation nor the private organizer's
gate. V25's eight fresh candidate journals equal v24's; its numerical result
reuses v24's fresh baseline replay on those identical histories, not a newly
run v25 baseline replay.

Two first-token distribution comparisons (248,320 logits) additionally found:
standard maximum error 1.06543, RMSE 0.18630, cosine 0.99525 and KL 0.00007653;
coding maximum error 0.90625, RMSE 0.16218, cosine 0.99624 and KL 0.00131370.
Both argmax IDs match, but coding continuation changes. These are limited
numerical evidence and do not justify default promotion.

Before cleanup, v25 passed 94 targeted native tests (one ignored real-checkpoint
harness), the complete 11-request checkpoint smoke with all 1365 blocks released,
254 forced decode replays (128 all-hit and 126 real misses), one forced wide
prefill replay, and all four exact coding comparison legs. Smoke covers AR,
streaming continuation, native/adaptive MTP, owner changes, cancellation recovery,
image continuation and replacement. Its smaller admitted plan had 366 slots.

The singleton PLE regression checks actual readiness before evaluating either
output, then exact F32/BF16 output and retained history across mixed singleton
and longer windows. A fresh CPU sample shows zero PLE event-wait samples versus
125/2282 model-thread samples before the fix (2557 samples afterward). This
confirms removal of the observed wait, not an end-to-end throughput gain.

Serialized profiles force stage completion and change scheduling. The initial
profile took 809.143 ms; ordinary execution of that build measured 701.416 ms.
A recent serialized profile took 618.668 ms, including experts 204.095 ms, GDN
projections 72.943 ms, attention branch 47.617 ms and recurrence 43.750 ms.
Warm windows had no expert misses, uploads, source reads or packed reads. These
intervals are neither ordinary throughput nor pure GPU durations.

Rejected or unpromoted research includes:

- FP16/TF32 operand substitutions that changed tens of thousands to over a
  million BF16 values across representative shapes; not equivalent storage ports.
- Wider dense NAX tiles and smaller singleton row groups without consistent gains.
- A wide router that matched IDs but changed normalized scores; only validated
  probability-first arithmetic is retained.
- GDN pairing across multi-token windows, which changed rounding; pairing is now
  restricted to singleton projections, with wider cached-pair fallback tested.
- Attention normalization using a different reduction tree; the retained optional
  port follows this checkout's reduction order and handles actual strided Q views.
- Metal 4.1 compatibility, removed after the four-process no-gain result.
- Vectorized fragment access: all 18 complete-output cases matched, including
  inactive experts and 1/15/16/17/31/32/33/65-row tails, but isolated changes ranged
  from -4.21% to +2.52%. The prototype was not integrated.

Full BF16 checkpoint generation, real-model contexts beyond 2048 tokens, audio,
video and image-bearing MTP remain unvalidated or unsupported as stated in the
runtime guide. Sparse thresholds are covered by small fixtures.

## Follow-up source audit, 17 September

Starting from the cleaned `f76ee4be` build, the renewed audit follows the
winning PR's source and mechanism list directly. Its four-row GDN schedule,
indirect expert tile table, packed NAX prefetch, split wide projection inputs,
route sorting, and first-two/then-three-layer submission schedule already
exist locally. The remaining exact candidates identified in this pass are:

| Reference mechanism                     | Local gap and port                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TrackFastKernels.leanRowsSource`       | The local four-row recurrence still used scalar loads, six pointer increments, and a scalar-array sequence length. `MLX_QWEN4_GDN_VECTOR_ROWS=1` uses vector loads, three shared offsets, and a template sequence length. It retains modulo head mapping, ordinary FP32 reductions, FP32 output, and FP32 state. The reference's Kahan arithmetic and consecutive head mapping are deliberately not copied into this GGUF path. |
| `TrackFastMoE` gate/up and down staging | Routed and shared gate/up activation writes were still serialized on lane zero. `MLX_QWEN4_EXPERT_LANE_STAGING=1` distributes complete reduced rows across lanes and independently enables distributed down-result staging at the existing four-row geometry. The older two-row schedule remains a separate choice.                                                                                                             |
| `TrackFastMixer.upMixSource`            | `MLX_QWEN4_MIXER_LANE_PRODUCTS=1` computes the four independent stream products in four lanes, then retains this checkout's ascending BF16 sum and division by four. It keeps the current weight layout and one-column SIMD ownership.                                                                                                                                                                                          |
| `TrackFastMixer.downInjectSource`       | `MLX_QWEN4_MIXER_DOWN_INJECT=1` fuses singleton mixer down/SiLU and injection projections into one dispatch, keeping the local affine8 lane walks and BF16 boundaries. It reads the existing two banks and adds no resident weights.                                                                                                                                                                                            |

These switches remain off pending repeatable full-model comparisons. The
recurrence regression checks every output and state bit for F32/BF16 inputs,
contiguous/strided/offset storage, modulo-tiled key heads, 7/9/1024-token windows and
one-token continuation. It passed on this host. Expert and mixer regressions
reuse ordinary projections and the native sigmoid halfway cases as independent
references. All 98 Qwen4 native tests passed on the complete three-port candidate (39.35 seconds).
The later down/injection fusion passed its independent projection regression,
including changed weights, offset views and unsupported-input fallback; the
expanded 99-test Qwen4 suite passed in 57.40 seconds.

A four-process, uncontended control/vector/vector/control check passed all
hashes and guards at the required 63 GiB/393-slot plan. Process medians were
1959.888/52.156, 1974.653/52.093, 1955.831/52.080, and 1932.010/52.212
(prefill/decode tokens/s). The small prefill change is insufficient to approach
90%; decode is unchanged. An isolated 1024-token recurrence probe matched
all outputs and final states exactly for both input dtypes. After its first
control block, FP32 control blocks took 1.463/1.444 ms versus candidate
1.382/1.394/1.404 ms; BF16 block times overlap. These kernel timings do not
replace the complete-request comparison.

The first same-binary combined check measured 1951.665/51.256 for its control
and 1962.355/52.313 for the three-port candidate (three measured requests per
process; 63 GiB/393 slots; hashes and guards passed). The control's last
prefill sample dropped to 1665.860, so this pair alone is insufficient for
promotion. Two subsequent optional-fusion runs stopped before loading because
another worktree was running Vitest. Those runs have no throughput result.
The later down/injection fusion is not included in this initial comparison.

With all three ports, the earlier optional fusions, BF16 prefill operands and
wide PLE enabled, an ordinary guarded run measured
2247.148/2250.736/2260.135 prefill and 53.851/53.764/53.403 decode tokens/s.
Its medians are 89.6%/85.0% of the published rates, still below both thresholds.
The standard output hash matched on every request; the prior BF16 numerical
limits still apply. This run also predates the down/injection fusion.

The following minimal-fusion run overlapped a `k2-horizon` Rust build and is
excluded from optimization comparisons. One-second telemetry reaches full
CPU utilization during that run. The guard now also rejects competing
mlx-node compilers; it had previously checked only model footprints and MLX
tests. This is evidence of contention in that particular run, not a measured
regression from disabling fusions.

A separate singleton GDN prototype ports the reference's first-state-row
prefetch and register-cached q/k values. It matched all three outputs for two
input phases. Warm control blocks took 153.474/153.187/153.607 microseconds;
candidate blocks took 155.159/150.237/150.881 microseconds. This small,
overlapping result did not justify integration in that pass. The original probe
remains in the external evidence archive; the later 95% follow-up tests an opt-in port. A follow-up 64-step chained probe
also matched every output, recurrent-state and history bit for two input phases.
Its timing ranges still overlapped (control 18.6–36.3 and prefetch 19.4–39.2
microseconds per step). A compiler was active during that diagnostic, so these
are not isolated GPU performance measurements or grounds for integration.

A separate CPU-side audit found repeated deep clones of the immutable model
configuration in forward helpers. The decoder now shares that configuration
through `Arc`, matching the reference's retained configuration without changing
any tensor or arithmetic. Its measurement is separate from the kernel ports.

The shared-configuration candidate passed all 99 Qwen4 tests and the broader
core release suite (3561 passed, 122 ignored, three existing debug-only
assertions excluded). Its full-checkpoint smoke passed all 11 requests and
released all 1365 pages and state reservations. Smoke used its actual
63 GiB/366-slot auxiliary-model plan and a 65.21 GB peak physical footprint.
This was correctness-only validation under the memory/pressure guard; other
small tests/builds were allowed, so its timings are not throughput evidence.
The canonical addon build and strict all-target Clippy also passed.

PR #154's previous-head CI exposed a compact-gate rounding difference in the
older-GPU compiled GDN fallback. Forcing that fallback on this M5 reproduced
it: the compact recurrent state differed in 203922 elements by at most
1.8626451e-9 at the first step. The fallback now promotes gate inputs to F32
before entering the compiled graph, restoring the original input signature;
the fused Metal path remains unchanged. A child-process regression explicitly
forces this fallback on every Metal host, with the existing exact output,
state and history assertions intact. Both the original replay test and the new forced-fallback regression passed
with zero differences in outputs, states and retained histories.

A later four-process candidate/control/control/candidate check retained the
same 63 GiB/393-slot plan, guards and standard hash. The two three-port control
processes measured medians of 2244.979/53.939 and 2238.188/53.818 tokens/s.
The down/injection plus shared-config candidate measured 1939.126/52.888 and
2207.549/53.514. The first candidate's prefill slowdown did not repeat at that
magnitude, but this comparison does not demonstrate a gain from the new fusion.
A preceding 20-leg same-process feature screen also passed all hashes and the
guard, but had large within-variant timing swings and cannot establish a winner.
Those samples are retained rather than selected for a favorable result.

A separate packed-mixer probe ports the reference's two-column threadgroup,
eight-product staging and reordered rows, with a control retaining the local
layout. All outputs matched bit-for-bit for two changing input/weight sets.
Repeated warm blocks overlapped: control 141.5–151.2 microseconds, unpacked
reference geometry 143.3–150.3, packed geometry 140.9–167.2. This includes
per-dispatch evaluation overhead; it is not a full-model gain. A follow-up
64-step chain used 64 distinct weight banks and dependent changing inputs.
Both reference geometries matched the final outputs for two weight/input
phases. Ranges still overlapped: control 12.74–16.15 microseconds per step,
unpacked reference geometry 12.91–15.79, packed geometry 12.69–13.49. No duplicate
mixer banks or associated residency change are added based on these results.

The next backend port follows the reference's source/compile-option hashes
as Metal library keys. `MLX_METAL_HASH_KERNEL_CACHE=1` selects it at process
start. The immutable key is computed with the custom primitive, avoiding
repeated source scans for retained compiled graphs. It is off by default and
has no established throughput gain yet. This uses the parent-owned Metal
overlay, with the same generated class header in CMake, the bridge and installed
headers. A compile-definition change invalidates old objects when that header
is first introduced. The initial incremental-build mismatch was caught by the
external correctness probe before any model benchmark; the rebuilt probe
passed 90 same-name source/options/strided-input comparisons and 100 changing
compiled calls in each cache mode. The cache-enabled broader release suite
passed 3562 tests with 122 ignored and the same three debug-only exclusions;
strict all-target Clippy passed. Its first full-checkpoint smoke was interrupted
before load completed when another task's K2 model exceeded 8 GiB; that attempt
provides neither checkpoint validation nor a throughput result.

The cache's four-process ABBA comparison passed all guards and standard hashes.
Control medians were 2272.210/54.574 and 717.399/35.424; cache-enabled medians
were 2255.924/54.688 and 743.040/33.489. GPU clocks fell from approximately
1.62 GHz in the early processes to 0.4–0.95 GHz in the later ones. AC High
Power mode remained enabled. This comparison is inconclusive; the slow repeated
control prevents attributing the decline to the cache implementation.

A further `TrackFastMoE` FULLTAIL audit found that compact dense decode still
used runtime-length helpers for its final packs. The native entry point admits
only K multiples of 32, so every participating tail lane owns a complete pack.
The port uses the ordinary fixed-size helpers inside the same lane guard,
preserving the walk, sum order, scales, biases and output rounding. The isolated
mixer comparison matched all output bits for two changing weight/input sets.
The permanent independent quantized-projection regression now includes
288/320/352/384-wide inputs, covering every admitted tail length of the normal
128-column walk. All 100 Qwen native tests passed on this build. Its contribution
has not been isolated from the later combined measurements.

The AVEC audit also found that `STAGE && !COMPACT` disabled the existing
prefetch when BF16 down-projection operands were selected. The port stages eight
BF16 values with aligned `uint4` loads/stores, matching
`TrackPrefillIndirectMetal.swift`'s AVEC implementation. The F32 staging retains
its existing conversions. The native entry point and padded rows guarantee
16-byte alignment. A child-process regression compares the staged BF16 path
against scalar loads for mixed gate4/5 and down5/8 banks, two changing inputs,
permuted token rows, and expert counts 0/15/16/17/31/32/33/63/64/65. All 101 Qwen
native tests passed. This proves equivalence of the load change within the
BF16 experiment; it does not make BF16 and the default F32 arithmetic equivalent.

The AVEC candidate/control/control/candidate comparison used the content-key
cache, reference RPS2 down schedule, earlier optional fusions, and identical
63 GiB/393-slot/1024-token admission. All four processes passed guards and
output hashes:

| Process                       | Prefill median | Decode median |
| ----------------------------- | -------------: | ------------: |
| AVEC candidate A              |       2339.489 |        55.409 |
| Scalar BF16 staging control B |       2275.841 |        55.327 |
| Repeated control C            |        706.664 |        37.326 |
| Repeated AVEC candidate D     |        841.309 |        42.804 |

The first candidate's prefill samples were 2339.489/2349.059/2338.355; decode
was 55.302/55.409/55.709. Early GPU clocks were approximately 1.62 GHz; later
samples included 0.56–0.58 GHz. The first pair suggests a staging improvement,
but the full comparison does not establish a repeatable gain. These observations
do not identify the cause of the clock change.

The follow-up port extends the reference's invariant inactive-row initialization
to down projections: each inactive stage slice is cleared once before the K
walk. `MLX_QWEN4_PREFILL_HOIST_ZERO=0` retains the repeated-zero control.
The reference's `MLXFAST-ROUTERRPS1` is also adapted to the local BF16 router:
one row per SIMD group, unchanged four-value K walk, F32 accumulation,
shuffle-down reduction and final BF16 rounding. It uses the already-accounted
BF16 weight cache and is selected by `MLX_QWEN4_ROUTER_ROW_SCHEDULE=1`.
The router matches native matrix multiplication bit-for-bit for four changing
input/weight sets, including offset and strided views; unsupported F32 inputs
fall back. All 102 Qwen native tests passed, including the BF16 staging/zero
comparison and the older-GPU fallback regression. The canonical addon build,
Rust formatting and strict all-target Clippy passed. The exact optional-fusion
configuration passed all 11 full-checkpoint smoke requests, including changed
images and continuation, and released all 1365 pages with no state reservations.
That auxiliary-model run used 63 GiB/366 slots and peaked at 65,242,335,344 bytes
of physical memory; its timings are correctness-only.

The 20-leg router schedule screen used two warmups and three measurements for
four variants. Median prefill/decode rates were 1985.536/54.366 for the prior
reference configuration, 2005.320/55.050 with router row scheduling,
1997.683/55.369 with the mixer down/injection pair disabled, and
2001.270/54.746 with the four-row expert down schedule. All hashes and guards
passed; GPU clocks varied within these runs. This screen motivated the final
configuration, but does not isolate a precise gain for each switch.

The final ROUTEREG port retains one winner per output lane instead of ten
winner values and indices in every lane. SIMD shuffles reconstruct the same
ascending ten-term score sum. Both singleton and prefill retain probability
rounding, the local tie rule, NaN handling and output order. Separate cached
graphs specialize the control and register variants; every changing tensor
remains an input. `MLX_QWEN4_ROUTE_REGISTER_RESULTS=1` selects this path. Its
child regression runs the existing independent probability/tie tests in both
prefill and decode; all 103 Qwen native tests passed.

The final content-key cache replay again passed all 190 same-name source,
compile-option, strided-input and changing-invocation checks in each mode.
The content-key cache also avoids hashing or allocating its derived key when
disabled. The process-start setting is captured when a custom primitive is
first constructed. Its enabled key and dispatch semantics are unchanged.

The final combined build and configuration produced the six above-threshold
samples reported at the start of this document. No per-port percentage is
inferred from that combined result. The final exact-arithmetic coding run
retained the expected `fa17b454...45a96` hash across five 1023-input/128-output
requests. Its three measured rates were 1099.361/1101.709/1085.064 prefill and
30.576/30.378/30.363 decode tokens/s. That distinct prompt and precision
configuration does not inherit the standard-workload 90% result. The faster BF16/PLE configuration was also run on that
coding prompt, with five deterministic requests and a distinct output hash,
`40d938a34d8106a2caed48e11fcaa08b15118668ad3d5cbac361466ddd5e9329`.
Its measured prefill was 1188.054/1181.745/1179.404 and decode was
30.100/30.224/30.423 tokens/s. These coding results are retained as workload
limits; the standard benchmark's 90% result is not generalized to them. A combined routing/shared-gate prototype
also matched 256 changing route/weight cases, but overlapping diagnostic timing
ranges did not justify another runtime variant; it remains outside the PR.

The first fresh default run was admitted at 386 expert slots rather than 393,
so it is excluded from fixed-budget comparisons (prefill
1951.314/1893.519/1666.196; decode 52.043/52.018/51.285 tokens/s). A subsequent
BF16 run overlapped another worktree's native MLX GPU suite and is also excluded
(prefill 905.610/859.015/874.646; decode 41.203/41.414/40.927). Its low rates are
not evidence of a BF16 regression. The external harness now requires exactly
63 GiB/393 slots/1024-token windows and stops on competing native MLX tests,
including tests below the existing 8 GiB memory threshold. It stops only its
own child group. This observed interference applies to these runs; it does
not retroactively establish the cause of earlier timing drift.

New local evidence lives in
`~/Library/Caches/mlx-node/qwen4-reference-gap-20260917/`, including immutable
control/candidate builds, invocation records, memory guards, telemetry, and
external probes. No harness or temporary file is added to the repository.

## PR cleanup and validation

Standalone benchmark, smoke, fixture-generator, memory-guard, profiling and
prototype sources, raw measurements and temporary binaries are archived outside
the repository. This is the single Qwen3.8 research report in the PR. Native
regression tests and their deterministic synthetic fixture data remain in-tree.
No new performance measurement is inferred from cleanup.

Two review findings are corrected: GGUF extension matching is case-insensitive,
and the advertised scheduler capacity returns one under `MLX_SERVE_FORCE_SERIAL`.
Their regressions exercise renamed GGUF fixtures through logits and isolated
processes reading the real scheduler policy. The TypeScript chat-family assertion
now includes `qwen4_exp`.

CI at `cb10e804` later failed two scheduled-owner tests with
`Qwen4 stopped weight growth to preserve system headroom`. Their MTP observation
path enters `prefill_chunk`, whose forward reserve check still queried the live
runner even for tiny fixtures. Explicit unit-fixture stores now skip that
forward-only reserve check under `cfg(test)`. Production stores still execute
it; actual payload admission and weight-growth checks are unchanged. An
injected failing check verifies both fixture isolation and production rejection. The production guard still runs at the same forward
boundary; this follow-up changes fixture isolation, not kernel arithmetic.

Before the fixture-isolation follow-up, the broader release suite passed
**3565 tests**, with 122 ignored and the same three debug-only exclusions. After
that follow-up, all **104 Qwen native tests** passed, including both previously
failing scheduled-owner tests and the injected production-headroom regression.
The final canonical addon build, packaged metallib checks and strict all-target
Clippy passed. The final addon is archived as `fixture-headroom-build`, SHA-256
`5d3d746a2b82186f583b94865789c142ffdbfd982e779146428f78f99481dfa4`.
Its production forward guard still calls the original check, and its kernel
sources match the measured `route-register-build`; performance was not rerun
after this fixture-only follow-up. A final BF16/PLE full-checkpoint smoke attempt was stopped during load
when the Devin process reached 8.80 GB, above the unchanged competing-process
limit. It supplies no additional smoke result. The earlier 11-request smoke,
the six above-threshold standard requests and both coding checks remain
separately identified above.

Tiny unit fixtures use an explicit bounded fixture store and injected bank-load
admission/refresh callbacks. They no longer request the production multi-GiB
working reservation from a busy CI runner. Actual payload reads still use live
checks; production plan/reserve behavior is unchanged, and planner snapshot tests
retain its insufficient-headroom coverage.

Earlier cleanup validation (17 September; later port checks are above):

- Canonical native addon build and packaged metallib smoke checks passed.
- Qwen native suite: 96 passed, zero ignored. Broader release core suite: 3558
  passed, 122 ignored; three existing debug-assertion tripwires were excluded
  because release builds disable their assertions.
- Full TypeScript run: 3750 passed, 40 skipped. Its six failures were all missing
  local GSM8K data; all eight dataset tests passed when retried with the existing
  dataset. The temporary dataset link was removed afterward.
- After rebuilding the addon, all 79 loader/registry/paged-policy/stream/agent
  tests across the five affected suites passed.
- TypeScript build/type checking, type-aware lint, strict all-target Rust Clippy,
  Rust formatting, scoped formatting/lint checks, and staged whitespace checks
  passed. Repository-wide `vp check` still encounters pre-existing formatting
  issues outside the PR; those unrelated files were not reformatted.
- Full-checkpoint smoke, numerical fidelity and performance were not rerun for
  this cleanup. Their pre-cleanup v25 evidence and limits are stated above.

Cleaned addon SHA-256:
`e5d6f3fb2cc2df130926abeac7ca7b9a5401728db66ccd690dc2931e149457be`.
The two packaged metallib hashes are unchanged from the v25 values below.
Remote CI is separate from these local checks.

## Evidence archive

Local evidence root:
`~/Library/Caches/mlx-node/qwen4-prefill-pass15-20260915/`.
The preceding audit remains at
`~/Library/Caches/mlx-node/qwen4-mlxfast-recheck-20260915/findings.md`.
These local artifacts are not committed and are not portable repository links.

The archive preserves exact plans, prompt IDs, token journals, per-leg samples,
output hashes, telemetry, guard records, failed runs, compiler invocations,
source snapshots and immutable native builds. Useful entry points include:

- `paired-default-final*`, `paired-coding-final*`, and the initial performance
  record under `cleanup-pr154-20260917/docs/research/`.
- `reference-v5-prefill-paired*`, `reference-v7-prefill-paired*`,
  `reference-v14*`, `reference-v16-final-paired*`, and `reference-v16-reuse-paired*`.
- `reference-v19-measurement.json`, `reference-v23-validation.json`,
  `reference-v24*`, `reference-v25-validation.json`,
  `reference-v25-ple-paired-summary.json`, and `reference-v25-fixed-recheck-summary.json`.
- `nax-vector-probe*`, `review-20260917-v25-final/`, its source patch/hash manifest,
  and `cleanup-pr154-20260917/removed.json` for the pre-cleanup archive inventory.

The pre-cleanup v25 addon SHA-256 is
`a5f47a8c0463d0bfd3ee6ede82993b85bb9267b1ba076aa23e0badb157112055`;
MLX metallib `7af3f85873830aecc1c516d2e7f7ed246d8e2aaacce95ee9ade765d2faf7a97c`;
paged metallib `bcc26de5b3cd65061a17a43fdd3ebe13e348d1e27caa3a5213f9d5c798df31d1`.
These identify the earlier correctness/performance evidence, not the subsequent
cleaned build. Ordinary timings must always be associated with their recorded
build, settings, workload and memory plan.
