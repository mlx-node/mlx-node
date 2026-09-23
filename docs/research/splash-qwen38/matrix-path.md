# Splash matrix-path experiments on this M5 Max

September 22, 2026. Target: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, with the supplied
BF16 DFlash2 companion. This round implemented and tested eight Metal matrix
prototypes. **None passed the performance gate, so none was enabled in the
runtime.** The existing validated implementation remains intact.

The last complete, matched-input local comparison remains
[35.6 / 41.4 tokens/s for mlx-node versus 64.9 / 78.9 for Splash](local-device.md),
for short / 6K prompts. Those are the preceding three-repeat, 1,024-output-token
measurements, not a new end-to-end run in this round. They use different weight
packages and slightly different decode timing boundaries. This investigation
does not establish a new throughput improvement or parity with Splash.

## What the source and profile showed

Splash is pinned to `7e3c67e8e3a9e9912ff6e02521457017cc4c65d0` in the local
`/Users/brooklyn/workspace/github/splash` checkout. Its unmodified
`dev/benchmarks/decode_profile.mm` was built and run on this device, with a
512-token prompt and four cycles per width. The single-request decode cycle
took 52.03 ms as one GPU command; separately timed dispatches summed to 54.39 ms.
Matrix projections accounted for 48.959 ms, about 90% of that isolated sum.
This diagnostic is a different workload from the full-request comparison.

The relevant implementation is
`runtime/metal/kernels/common/q4_mpp_tiles.h`, selected by
`runtime/metal/kernels/decode/linear_q4.metal` and `runtime/ops/Linear.cpp`.
Splash passes packed integer weights directly to Metal's cooperative matrix
operation, then applies each quantization group's scale and bias to the FP32
partial. Its paired variants issue two group operations before their epilogues.

Our existing M=8 verifier uses `qmv_wide` in
`crates/mlx-sys/mlx/mlx/backend/metal/quantized.cpp` and
`kernels/kquant.h`. It already decodes a weight chunk once and reuses it across
eight activation rows. Replacing it must beat that reuse, not an eight-pass
matrix-vector baseline.

The requested GGUF has mixed formats. Header-only inspection of its generated
native cache found these packed matrix payloads, excluding embeddings and
one-dimensional weights:

| Format        | Matrices | Packed weight bytes |
| ------------- | -------: | ------------------: |
| Q5_K          |      191 |       7,205,683,200 |
| IQ4_XS        |       70 |       2,941,255,680 |
| Q6_K          |       56 |       2,616,852,480 |
| Q4_K          |       68 |       2,084,044,800 |
| Other formats |      120 |         561,643,520 |

These are code payloads, not total model residency: scale arrays, biases, dense
weights, caches, activations, and the draft are additional.

Q4_K/Q5_K/IQ4_XS have scale groups of 32; Q6_K uses 16. Splash's affine Q4
uses 64. Combining two independently scaled GGUF groups into one K=64 dot and
applying one scale afterward would be incorrect.

## Executed plan

1. Verify Metal's integer operand support before changing the loader or runtime.
2. Start with Q5_K, the largest payload, preserving all codes and parameters.
3. Compare expanded bytes, native packed input, and tiled packed input; test
   four-way K splitting, four/eight SIMD groups, and paired group operations.
4. Test Q4_K's direct packed-four-bit path, which avoids Q5's byte expansion.
5. Use a high-precision reference, the current native kernel, repeated timing,
   and independent source review. Integrate only after a substantial kernel
   win, then build the addon and run full-model A/B and lifecycle validation.

Step 5 rejected every candidate before runtime integration. There was no reason
to subject the full model to kernels already slower on all tested shapes.

The installed compiler accepts BF16 × uint8/int8 → FP32 at K=16/32/64, and
BF16 × packed uint4 → FP32 at K=32/64, for both device and threadgroup weights.
It does not expose the probed packed uint2/uint5/uint6 types. Compile success
was only a feasibility check; the runtime tests below established behavior.

## Repeated results

All variants process eight activation rows. Each of three fresh processes per
variant tests nine shapes: a small full-reference fixture, the seven matrix
shapes present in the Q5/Q4 inventories, and a merged gate/up-sized projection.
Each shape has 32 warmup pairs followed by 15 alternating timed pairs.

Both arms use the same Metal queue and command-submission boundary. The
baseline directly invokes the frozen production `qmv_wide` specialization;
its output must exactly match MLX's `quantized_matmul` output before timing.
GPU timestamps cover the whole command, including the second reduction
dispatch for split-K candidates. Preparation/repacking is excluded.

The table reports median per-process GPU speed ratios. **Below 1.0 is slower
than the existing kernel.**

| Candidate                                                | N=17,408, K=5,120 | N=5,120, K=17,408 | Decision |
| -------------------------------------------------------- | ----------------: | ----------------: | -------- |
| Q5 expanded bytes                                        |            0.570× |            0.259× | Reject   |
| Q5 expanded, four K partitions                           |            0.700× |            0.731× | Reject   |
| Same, four SIMD groups                                   |            0.597× |            0.540× | Reject   |
| Q5 native packed, unpack in threadgroup, four partitions |            0.452× |            0.456× | Reject   |
| Q5 tiled packed, four partitions                         |            0.525× |            0.475× | Reject   |
| Q5 expanded, four partitions, paired operations          |            0.683× |            0.633× | Reject   |
| Q4 packed input, four partitions                         |            0.741× |            0.684× | Reject   |
| Q4 packed input, paired operations                       |            0.731× |            0.688× | Reject   |

For the best Q5 variant, the individual projection took about 0.285 ms versus
0.200 ms for the existing kernel; the down projection took 0.286 versus
0.208 ms. Q4's individual projection took 0.250 versus 0.185 ms. Every
shape/candidate/process ratio was below 1.0, well short of the 1.25× continuation
gate. These are operator measurements, not generated-token throughput.

A model-wide Q5 byte sidecar would also be expensive. Keeping the original
packed weights for AR/prefill adds about **11.529 GB of expanded codes plus
2.882 GB of FP32 parameters**, approximately 14.411 GB. The smaller 4.324 GB
figure is only the expansion delta when replacing, rather than retaining, the
original packed payload. No model-wide sidecar was allocated.

## Correctness and review

The final cohort completed **24 successful processes and 216 shape checks**.
The harness verifies:

- Exact native-kernel versus MLX output identity for the baseline.
- Full output finiteness and coverage, with NaN sentinels in output/partial buffers.
- CPU float64 reconstruction and accumulation for every small-fixture output
  and 32 distributed outputs per large shape.
- Maximum, p99, RMS, normalized RMS, thresholded relative error, and mismatch
  counts against correctly rounded BF16 references.
- Candidate normalized RMS no more than 1.10× baseline, and maximum error no
  more than the larger of 2× baseline error or two BF16 ULPs at the fixture's
  maximum magnitude.
- Exact candidate output stability across repeated execution.

Fixtures contain varied five/four-bit codes, scale/min extremes, non-power-of-two
FP16 super-scales, and signed BF16 inputs. The candidates passed these bounded
numerical checks. Some outputs differ from the current kernel at BF16 rounding
boundaries, so this is not a bit-exact or end-to-end transcript-parity claim.
Large-shape CPU references are sampled, and these are synthetic weights with
real model dimensions. Operator buffers are warmed and repeatedly reused; the
screen does not rotate through the full model's weight working set. Since every
candidate failed the speed gate, no
full-model candidate generation or broader quality claim was made.

GPT 5.6 Sol subagents independently investigated the verifier and recurrent
state paths, compiled operand probes, and reviewed packing, scale/min indexing,
cooperative output traversal, timing boundaries, the reference calculation,
and Q4 adaptation. Review corrections included a same-queue baseline, stronger
fixtures, true float64 reconstruction, sentinels, correct sidecar accounting,
and the exact BF16-ULP gate. All were incorporated before the final cohort.

The standalone harness builds with warnings treated as errors; warnings in
preexisting MLX headers are locally suppressed around those includes. Shader
builds use Metal 4.0, `-O3 -fno-fast-math -Wall -Wextra -Werror`. The existing
runtime patch and MLX-submodule patch are byte-identical to their pre-round
snapshots. The current addon and frozen validated addon both retain SHA-256
`9f1e274b9178a410bb8cc3d967d4c54b80e672bd4297d1be2c2e2bec064469f4`.
Runtime test suites were not rerun for unaccepted, standalone experiments.

## Other findings and remaining work

State snapshots already alias their buffers, and accepted-prefix GDN replay is
already fused per layer. A verifier state tape would require a second recurrence
to preserve serial BF16 rounding, plus hundreds of MiB of writes per cycle.
That is not a free replacement for replay. Device-count commit scheduling is a
smaller, still unmeasured opportunity; its benefit should not be presented as a
route to closing the measured gap by itself.

The M5 Max capability probe reports timestamp counter buffers and stage-boundary
sampling, but **no dispatch-boundary sampling** through the current Metal API.
Adding the proposed per-dispatch counter calls would therefore be unsupported.
Splitting encoders to obtain isolated stage timings would perturb synchronization
and must be treated as an intrusive diagnostic.

The evidence rejects these implementations, not every possible integer-matrix
kernel. Further work needs either a measured improvement to the original packed
GGUF arithmetic/layout, or a larger execution-path change whose benefit survives
full-model validation. Merely adopting Splash's matrix API is insufficient for
this checkpoint. Changing to Splash's affine-Q4 package would be a different
model comparison, not an optimization of the requested GGUF.

## Artifacts

Everything is retained under `.cache/benchmarks/splash-qwen38-phase3/`:

- `mpp-summary.json` and `summarize-mpp.mjs`: validated final tables and raw rows.
- `mpp-reviewed-*.log` / `.job.json`: the 24 accepted final processes.
- `mpp-runtime/`: shader sources, compiled libraries, harness, and build script.
- `mpp-probe/`: reproducible compile probes and expected unsupported-type failures.
- `splash-profile.log`: unmodified Splash's local operator profile.
- `format-inventory.json`, `q5k-shape-inventory.json`, `q4k-shape-inventory.json`.
- `counter-capabilities.mm` / `.log`, research notes, and `manifest.json`.

Earlier smoke/compile failures are excluded from the final summary. Operator
jobs finish before the guard's five-second RSS poll, so their recorded zero
maximum RSS means unobserved, not zero consumption. Allocations are bounded to
individual matrices; the larger Splash profile has resource samples. Final-cohort
GPU jobs ran sequentially, without concurrent builds; desktop activity and GPU
clocks were not controlled.

Example from the repository root:

```sh
.cache/benchmarks/splash-qwen38-phase3/mpp-runtime/build-harness.sh \
  .cache/benchmarks/splash-qwen38-phase3/mpp-runtime/q4_mpp_split.metallib q4_split
.cache/benchmarks/splash-qwen38-phase3/mpp-runtime/mpp-runtime-harness \
  .cache/benchmarks/splash-qwen38-phase2/final-validated/mlx.metallib \
  .cache/benchmarks/splash-qwen38-phase3/mpp-runtime/q4_mpp_split.metallib q4_split
node .cache/benchmarks/splash-qwen38-phase3/summarize-mpp.mjs
```
