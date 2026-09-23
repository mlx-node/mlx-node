# SIMD and memory execution follow-up

Research-only source audit, September 22, 2026. No runtime files, weights, builds,
or GPU jobs changed. Scope is the existing
`.cache/models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf` and BF16 DFlash2
companion. Splash was read at clean commit
`7e3c67e8e3a9e9912ff6e02521457017cc4c65d0`. Paths prefixed `splash/` below are
relative to `/Users/brooklyn/workspace/github/splash`; other source paths are
relative to `/Users/brooklyn/workspace/github/mlx-node`.

Read [architecture-implementation.md](architecture-implementation.md) and
[matrix-path.md](matrix-path.md) first. Revised segmented attention remains
enabled; residual/SwiGLU QMV fusion remains opt-in. These notes identify
hypotheses, not demonstrated gains or a claim that bandwidth explains the gap.

## What is already present, and what Splash actually adds

| Property                          | mlx-node evidence                                                                                                                                                                                          | Splash evidence                                                                                                                                        | Implication                                                                                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Weight reuse across verifier rows | `crates/mlx-sys/mlx/mlx/backend/metal/quantized.cpp:452-479` selects one input-row tile for M=8, N>=2048. `kernels/kquant.h:1107-1129` decodes eight weights once, then consumes them for every input row. | `splash/runtime/metal/kernels/common/q4_mpp_tiles.h:102-106,143-160` forms an 8-by-N cooperative matrix tile.                                          | Both already reuse a weight across eight verifier rows. Replacing eight independent vector passes is not an available optimization here.                             |
| Output reuse domain               | `kernels/kquant.h:1080-1105`: eight K lanes per output, four outputs per SIMD group, two SIMD groups. Each thread owns one output's eight accumulators.                                                    | `q4_mpp_tiles.h:102-120`: 128/256 output columns per cooperative tile and group-contiguous weight layout.                                              | A wider output reuse domain is a useful architectural difference; copying the cooperative API already failed.                                                        |
| Activation loads                  | `kernels/kquant.h:1121-1126` loads each activation again for each output row's dot product at the source level.                                                                                            | `q4_mpp_tiles.h:65-85,140-185` shares activation sums in threadgroup memory; the matrix operation consumes the input slice across many output columns. | Source duplication is an instruction/reuse hypothesis, not a measurement of DRAM transactions.                                                                       |
| Latency overlap                   | Current QMV has one scale/decode/consume loop (`kquant.h:1109-1130`).                                                                                                                                      | `q4_mpp_tiles.h:88-92,188-210` issues two group matrix operations before their epilogues, retaining group accumulation order.                          | QMV-specific prefetch/load scheduling remains distinct from the already-rejected paired MPP experiment; extra live registers can erase its benefit.                  |
| Persistent output traversal       | QMV launches one group per output tile (`quantized.cpp:476-480`).                                                                                                                                          | `splash/runtime/metal/kernels/decode/linear_q4.metal:18-24` strides output tiles with persistent groups.                                               | This does not retain an entire layer's weights or avoid a subsequent layer's weight stream. It changes scheduling and input locality, with uncertain benefit to QMV. |
| Dynamic selection                 | Existing QMV route contains architecture and shape heuristics (`quantized.cpp:208-249,425-469`). New epilogue legality checks inspect compiled pipeline properties (`kquant_epilogues.h:14-27`).           | `splash/runtime/ops/Linear.cpp:192-243,253-293` uses measured family/core constants; candidates at `342-345` include literal group counts 36/60/80.    | Borrow legality/measurement separation and workload-keyed choices, not Splash's machine-specific constants.                                                          |

All `kernels/kquant.h` citations above mean
`crates/mlx-sys/mlx/mlx/backend/metal/kernels/kquant.h`.

The native storage is already a lossless GGUF repack into code, sub-scale and
super-scale arrays; it is not raw GGUF blocks fed directly to Metal.
`crates/mlx-core/src/utils/gguf_kquant.rs:1-50,147-170` defines the contract,
chunked importer and reference gate. Changing byte layout is permissible in
principle without changing quantization, but requires a new explicit storage
contract and correct AR/prefill consumers or bounded sidecars.

QMV's exact arithmetic is materially different from Splash's: native QMV
decodes weights to FP32 first, accumulates each eight-value subchunk in order,
adds those subchunk partials in order, then uses the fixed K-lane shuffle ladder
and casts to BF16 (`kquant.h:1117-1156`). Splash computes an integer-code dot,
then multiplies by scale and adds bias times an input sum
(`q4_mpp_tiles.h:162-178`). Algebraically distributing those terms in native QMV
changes rounding. Preserve the current operation order in an exact experiment.
Q5/Q4/IQ4's group width 32 and Q6's width 16 cannot be merged into Splash's
single scale per 64 values (`gguf_kquant.rs:87-93`).

The BF16 draft does not use this quantized decoder for its dense projections.
`crates/mlx-core/src/models/qwen3_5/dflash2.rs:286-297` has separate gate/up/down
projections; `crates/mlx-core/src/nn/linear.rs:60-99` sends ordinary linears to
the dense matmul path. QMV improvements target the mixed-GGUF verifier and
applicable shared head work, not every draft projection. Splash's Q4 draft
does not establish a BF16 draft kernel opportunity by itself.

## Hypotheses that differ from the rejected work

### 1. Aligned native Q5/Q6 loads and exact integer extraction

Start here because it can preserve both storage size and the FP32 operation
sequence. Current Q5 extraction assembles straddling codes from byte operands
(`kquant.h:590-602`); Q6 does likewise (`606-613`). A Q5 group has 32 codes in
20 bytes, and a Q6 group has 16 codes in 12 bytes. Under the established row
and group alignment, these are five or three aligned 32-bit words respectively.
Load a group's words once and extract the identical integer codes using
constant shifts/masks, consuming the same eight-value subchunks in the same
order. Validate every code, including crossing-word fields and the last group;
no speculative load beyond the logical buffer. Do not change floating scale,
bias, multiply/add contraction or accumulation ordering.

This is not the rejected byte-expanded or tiled MPP kernel. It keeps QMV,
K-lanes, input-row reuse, packed bytes and reduction topology. It may do
nothing: the compiler could already combine byte loads/common expressions,
or extra live words could increase register demand. Inspect generated shader
instructions/resource reports where tooling permits; compare byte-load and
word-load variants with otherwise identical kernels. Q5 is the first workload
because it is the largest recorded packed projection payload, not because its
measured runtime fraction is already known.

### 2. Activation-load ownership within a SIMD group

At K-lanes=8, lanes `k`, `k+8`, `k+16`, `k+24` handle four output rows and read
the same activation for each `(input row, group, subchunk, element)`.
Prototype two otherwise identical QMV bodies:

1. Current direct loads from every lane.
2. One owner per K lane loads the activation; `simd_shuffle` distributes it to
   the other three output-row lanes, using the same source lane mapping for
   every SIMD group. All lanes participate; only the owners issue loads.

Keep float conversion, dot accumulation, output ownership, launch size and
weight bytes fixed. First broadcast one element at a time, preserving each
row's eight additions, so the experiment does not retain an entire M-by-8
activation block. Add a register-pressure-controlled chunk variant only if
the element version supports the hypothesis.

This tests load instruction issue/address/conversion pressure against shuffle
cost. It does **not** promise four times less DRAM traffic: identical-address
loads may already broadcast/coalesce; cache hits can satisfy repeated groups;
masked owner loads still require instructions and shuffles. Direct loads may
be faster. Report actual memory traffic only if a supported measurement exposes
it. A faster shuffle variant with unchanged measured traffic would support an
instruction/latency explanation, not a bandwidth claim.

An alternative is two output rows per thread with one activation load feeding
both dots. This avoids shuffles but doubles persistent output accumulators and
may retain both decoded chunks. It also reduces threadgroup count at fixed
threads. Keep it second-line: separate register/live-range cost from output
tile width and test enough N to expose lost parallelism. This is not the old
SG4/SG8 change, which only regrouped the existing output ownership.

### 3. QMV-specific, byte-preserving layout and software pipelining

Only pursue if the preceding diagnostics expose load/address stalls. Native
storage is row-major (`kquant.h:1093-1098`), while Splash groups output rows by
quantization group (`q4_mpp_tiles.h:109-120,148-151,169-170`). Try a small
output-row/group-block interleave matched to the existing QMV lane mapping,
retaining each code and original scale bytes. Start with a single matrix;
do not generate a model-wide second copy. The current mapping already accesses
contiguous groups within each output row, so improved transaction efficiency
is not guaranteed and transposition might make it worse.

The negative phase-3 Q5 tiled prototype already used
`[N/128,K/32,128,20]`, expanded FP32 scale/bias, threadgroup byte unpacking,
cooperative matmul and split-K (`.cache/benchmarks/splash-qwen38-phase3/mpp-runtime/q5_mpp_tiled_split.metal:7-17`).
Therefore “tile weights like Splash” is not novel. The untested separation is
native-QMV arithmetic plus a byte-preserving layout, no cooperative matmul,
no split-K, no metadata expansion. Test layout alone, then load scheduling;
otherwise any result cannot identify its cause.

A one-chunk look-ahead may overlap weight/scale loads with the previous chunk's
dot products. Retain accumulator update order, and compare no-look-ahead and
look-ahead at the same topology/layout. This borrows the latency-overlap idea,
not Splash's paired-MPP implementation. Extra weight/activation registers and
instruction-cache growth from unrolling are explicit rejection risks.

Do not first precompute all FP32 scales and biases. For Q5, current native
storage is 20 code bytes + 2 sub-scale bytes + 0.5 amortized super-scale bytes
per 32 values. Replacing metadata by two FP32 values increases 22.5 to 28 bytes
per group, about 24.4%, before any duplicate layout. Removing a small amount
of scale arithmetic can lose to this larger stream. These are layout bytes,
not observed memory traffic. `kquant.h:678-695` locates the original arithmetic.

### 4. Runtime measurement for legal, semantically equivalent variants

Autotuning is a selector, not a substitute for a winning kernel. Do not
reintroduce all negative variants and expect a different policy to create a
gain. Initially select between the established baseline and one candidate
that passed exact correctness and rotating-weight/operator tests.

Useful Splash design: `splash/dev/tuning/LinearTuning.cpp:191-210` rotates
representative weights, includes complete dispatch chains and records GPU and
wall time separately; `218-261` checks every representative and repeated
scratch reuse bit-for-bit; `splash/dev/tuning/Tuning.cpp:43-110` checks alternating
orders, noise and conservative paired gains. These tests still do not prove
that a warmed representative ring matches full-model cache behavior, explicitly
acknowledged at `LinearTuning.cpp:231-234`.

Compute legality from actual pipeline `threadExecutionWidth`,
`maxTotalThreadsPerThreadgroup`, static threadgroup memory, device maximum
threadgroup memory and proposed dynamic scratch. Include shape/tail alignment,
dtype, quant mode/group size and exact layout version. A kernel algorithm may
require width 32 and a fixed reduction topology; those are verified constraints,
not device-speed constants. Reject unsupported candidates without executing
them. Existing pure-profile checks in `kquant_epilogues.h:14-27` are a template
for synthetic capability tests, including unavailable/insufficient properties.

These runtime properties establish legality only. The maximum threads value
is an upper bound, not achieved occupancy, available registers per lane, active
SIMD groups, a GPU-core count or sustainable bandwidth. Actual register counts,
spills, issue stalls and cache/DRAM traffic are not furnished by that existing
capability contract. Use compiler/profiler evidence only when available;
otherwise label register-pressure explanations as hypotheses and use controlled
comparisons. Do not derive “optimal resident waves” from max threads alone.

Generate any exploratory grid choices from workload tile counts and legal
threadgroup sizes, using a bounded geometric subset plus the current baseline.
Do not import Splash's group counts or assume a runtime-reported core count
exists. Bound candidate count, scratch bytes, elapsed calibration time and
representative count; abort to baseline on pressure/noisy results/cancellation.
Use actual existing weight views rather than model-wide clones. Tune outside
the live generation critical path, with explicit ownership and no concurrent
benchmark. Cache a choice by device identity/capabilities, OS/compiler/library
identity, semantic/layout version and workload signature; invalidate on drift.
Device identity scopes measured evidence, never selects an invented speed table.

## Measurements that distinguish the limits

The recorded counter probe exposes only `timestamp`, stage sampling true and
dispatch sampling false
(`.cache/benchmarks/splash-qwen38-phase3/counter-capabilities.log:1-4`). It is not
a record of cache/DRAM bytes, occupancy or ALU utilization. Re-query supported
features on each device. A future supported profiler may expose more, but no
proposal should assume those counters or dispatch-boundary sampling exist.
If diagnostic encoder splitting is required, report it as intrusive and retain
a separate uninstrumented cohort. Command-envelope union and bound-buffer
capacity from the current command trace are not shader busy time or traffic.

| Possible limit                        | Controlled comparison                                                                                                                                                                                                         | Evidence that supports it, with limits                                                                                                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| External weight bandwidth             | Same kernel/shape, repeated matrix versus a bounded rotation of different matrices; increase resident rotation within the memory guard, not via a cache-flushing allocation. Compare supported traffic counters if available. | Timing growing with weight working set plus high measured external traffic supports cache/bandwidth sensitivity. A ring timing plateau only establishes stability for that tested ring, not proof of cold DRAM. |
| Packed-code/integer issue cost        | Q5/Q6 byte extraction versus aligned word extraction with identical logical bytes, dot order, grid and M. Compare hot and rotating cohorts.                                                                                   | Stable speedup without reduced traffic supports an instruction/load-path improvement. Similar hot/cold ratios alone do not identify exact ALU stalls.                                                           |
| Activation load/address issue         | Direct loads versus owner-load plus shuffle, fixed weights/output grid and live state. Run M=2..8 and several N/K shapes.                                                                                                     | Gain growing with M at similar traffic is consistent with duplicated load/conversion work; shuffle regressions reject this ownership scheme, not every wider tile.                                              |
| Register pressure/latency hiding      | One versus two output accumulations per thread, and look-ahead off/on; include shader resource/spill reports when obtainable.                                                                                                 | Spills or lower measured occupancy strengthen the hypothesis. Lower runtime max-threads alone is not an occupancy measurement; no-report inference must remain qualified.                                       |
| Insufficient independent work / tails | Vary N at fixed K/M and preserve exact output math; compare one versus multiple streamed output tiles only after a kernel win.                                                                                                | Shape-specific gain/loss locates crossover empirically. Do not turn one device's crossover into a universal N threshold.                                                                                        |
| Launch/submission overhead            | Same dependent chain encoded as production; record GPU command time and request wall time separately.                                                                                                                         | Kernel wins that disappear in the chain are not deployment wins. Existing zero-dispatch removal did not establish a throughput gain.                                                                            |

Use the existing kernel's unique packed codes + scales + super-scales per
input-row tile as a logical payload accounting model; there are
`ceil(M / vecs_per_tg)` such tiles. This is not actual load count: for example,
several lanes reference the same super-scale. Nor is it a DRAM lower bound,
because caches may retain payload. For the target's wide M=8 route there is
one input-row tile, not eight. Count output/input/scratch separately. Source-level
activation references scale as M*N*K, but turning that into external bytes
ignores SIMD broadcast/coalescing and caches. “Model payload / time” is an
effective payload rate, not measured DRAM bandwidth; advertised device bandwidth
does not measure this workload. Keep accepted tokens per verifier cycle when
relating cycle time to emitted-token throughput.

## Bounded implementation order and stopping rules

1. Freeze source/addon/library identities and the default baseline. Reuse the
   exact operator baseline qualification from phase 3. Capture format/shape/M
   inventory and actual dispatch choices; do not load another full model.
2. Prototype only Q5/Q6 aligned integer extraction in a standalone single-matrix
   harness; exact integer, FP32 decode and BF16 output tests, including buffer
   tails and pathological scales. Inspect resources, then alternating same-queue
   timings. Keep baseline if no gain. No loader/storage change at this step.
3. Independently prototype owner-load activation broadcast against direct
   loads. Hold launch and math fixed. Retain only a gain exceeding observed
   paired noise in both reused and rotating weights. Do not call logical load
   reduction measured bandwidth savings.
4. Only if measured evidence identifies a residual load/layout issue, prototype
   one bounded QMV-specific lossless layout. Include repack time, duplicate
   residency and AR/prefill fallback cost in the promotion decision. A verifier
   sidecar that requires another full model is not automatically acceptable.
5. Integrate a capability-checked selector for only accepted variants. Validate
   synthetic unsupported devices, batch/stride/tail fallbacks and cache
   invalidation. Preserve explicit matrix-route overrides and baseline numerical
   semantics. Do not add current-device hardcoded tuning values.
6. Run complete dependent MLP/projection chains with rotating model weights,
   then alternating fresh-process short/6K/32K full-model runs with fixed target,
   BF16 draft, output hashes, acceptance/cycle records and memory guard. Validate
   streaming, stop clamp, cancellation and cached continuation. Segment attention
   remains at its validated default. No improvement claim before this gate.

Not new experiments: larger SG count alone, K-lanes 16/32, forced M=8 split-K,
corrected BM16 NAX, eight expanded/native/tiled/paired integer-MPP variants,
or simply enabling residual/SwiGLU epilogues. Their negative evidence is in
`follow-up.md:32-57`, `matrix-path.md:90-117`, and the other-candidate screening
in `architecture-implementation.md`. Existing compatible projection merges,
eight-row weight reuse and segmented attention should not be rebuilt as if
absent. There is no evidence here that the proposed bounded changes can close
the full Splash gap; the experiments isolate plausible remaining instruction,
layout and scheduling costs while retaining the requested numerical contract.
