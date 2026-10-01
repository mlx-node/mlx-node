# Remaining performance controls

Original audit: September 22, 2026, against candidate 16 on
`codex/splash-qwen38-performance` (base `377e78f3`). Scope: the Qwen3.8-27B
native GGUF target, native BF16 or then-supported imported Splash Q4 DFlash2
companion, and shared Metal execution paths.

Cleanup status, September 23: the imported packed-Q4 loader and DFlash depth
override are removed. Retained eligible optimizations no longer expose manual
rollback switches; Q8 target-head reuse is unconditional within Q8, while BF16
remains the default. The earlier [cleanup](cleanup.md) removed only five phase-6
experiments; this cleanup also addresses controls from earlier phases.

The retained evidence below did not demonstrate a general speedup left disabled
by the benchmark. It does not prove that each candidate is always slower.
Full end-to-end screening and final cleaned-binary validation are complete;
[cleanup-final.md](cleanup-final.md) records those decisions and the reversed
12-request comparison. Exact output/acceptance is preserved, with no established
new performance gain.
Earlier measurements used their documented binaries and are not new results
from the current source tree.

## Removed scheduling, replay and column experiments

The eager-commit submission hook, direct-convolution replay shortcut and
conditional pressure-cache trimming are removed. The E48 four-column override
is also removed for this Qwen path; Qwen4's separate four-column route remains.
These are removed candidates, not disabled optimization choices. The retained
baseline asynchronously submits target state at normal continuation boundaries,
publishes lazy draft context at finish, replays convolution history, and clears
allocator cache at the existing periodic checkpoints.

Historical measurements from the earlier cohorts remain:

- Eager commit: off/on medians 33.91/35.44 short and 39.03/40.79 6K tokens/s,
  with varying pair direction and an effectively tied final pair. Earlier
  correctness checks passed; those are not validation of the cleanup binary.
- Direct convolution: 33.84/37.71 short/6K tokens/s, between baseline-before
  33.21/37.66 and baseline-after 34.03/39.07; no established improvement.
- Conditional cache trim: 33.22/37.54 tokens/s; retained allocator cache rose
  from about 1.23 GB to 7.70 GB without an established throughput gain.
- Four-column override: exact fixtures, but dependent-chain calls improved
  27.832→26.905 microseconds while synchronized calls regressed
  229.544→238.320; no established model-level win.

See [cleanup-final.md](cleanup-final.md) for the phase-8 decision evidence and
its limitations, completed cleaned-binary validation and final paired results.

Evidence: [candidate screening and limitations](architecture-implementation.md),
[state scheduling raw results](../../../.cache/benchmarks/splash-qwen38-phase4/state-summary.json),
[residual repeat](../../../.cache/benchmarks/splash-qwen38-phase4/ablation-residual-2-comparison.json),
[SwiGLU](../../../.cache/benchmarks/splash-qwen38-phase4/ablation-swiglu-comparison.json),
[convolution replay](../../../.cache/benchmarks/splash-qwen38-phase4/ablation-conv-comparison.json),
[empty finalization](../../../.cache/benchmarks/splash-qwen38-phase4/ablation-empty-comparison.json),
[cache trimming](../../../.cache/benchmarks/splash-qwen38-phase4/ablation-pressure-comparison.json),
and [recurrent-kernel screening](follow-up.md).

## Removed epilogue and submission experiments

The K/IQ residual and SwiGLU epilogue implementations and their switches have
been deleted, as has the empty-finalize shortcut. These are removed candidates,
not disabled optimization choices. Retained historical measurements:

- Residual's reversed repeat gained only 1.65% / 1.07% on short / 6K; its
  operator result was effectively tied.
- SwiGLU slowed full-model decode 4.26% / 7.73%.
- Empty finalization reduced submissions but measured 33.51 / 37.62 tokens/s
  in the earlier screening cohort, without an established throughput gain.

The phase-4 raw comparisons linked above and
[architecture implementation report](architecture-implementation.md) preserve
those measurements. They are not results from the current cleanup binary;
remaining phase-8 decisions belong in [cleanup-final.md](cleanup-final.md).

## Removed native Metal chunked GDN

The native Metal chunked-prefill implementation and its `chunked` selector have
been removed. The source-frozen `screen-gdn-chunked-3` observation belongs to
cohort A: one process measured all three prompts against the preceding base-d,
without a following baseline. Outputs and acceptance changed. This is a
screening observation, not a repeated paired estimate or final-binary validation;
see [cleanup-final.md](cleanup-final.md) for timings and the decision record.

The separate device-agnostic/CUDA `chunked_ops` path remains, including direct
log-space decay computation and numerical tests. Eligible long, unmasked
non-Metal prefill selects it by default; `MLX_GDN_KERNEL=perstep` selects the
serial ops reference and `chunked_ops` retains the chunked route. Neither
setting activates a chunked Metal verifier. The eight-row DFlash verifier
continues to use per-step recurrence.

## Conditional precision and adaptive options

| Option             | Current behavior                                                                                                                                                    | Retained evidence and limitation                                                                                                                                                                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Draft precision    | Fixed, no switch: every dense draft projection loads as affine Q4/group64 and the draft reuses the target head. The selector projection keeps checkpoint precision. | Chosen over BF16 and Q8 by teacher-forced acceptance on fixed BF16 transcripts (every arm within about 1.5 % of BF16) and clock-matched decode time: about 3.4 % less time per committed token than BF16 and 1.8 % less than Q8/group64, with 2.40 GiB less resident memory. Precision changes alter proposals and verify grouping, so transcripts differ from BF16. |
| `mtpAdaptiveDepth` | Ignored for Qwen DFlash2; its adaptive fallback has been removed. Native MTP and other draft families retain their own policies.                                    | Earlier Qwen adaptive pilots are historical, not available optimization choices. See [cleanup-final.md](cleanup-final.md) for decision evidence and validation status.                                                                                                                                                                                               |

Current sources: [draft loader/head policy](../../../crates/mlx-core/src/models/qwen3_5/dflash2.rs)
and [parameter resolution](../../../crates/mlx-core/src/models/qwen3_5/model/chat_backend.rs).

### Removed conditional experiments

The following results belong to historical binaries, not available runtime
switches or checkpoint choices:

- **Q8 head clone versus reuse:** reuse saved about 1.332 GiB active and
  2.100 GiB peak MLX allocation. The 256-token pilot improved Q8 throughput
  8–12%, partly through acceptance changes, but Q8+reuse was 12.3% / 34.5%
  slower than BF16 for 1,024-token short/6K responses. Memory is the reliable
  benefit; current Q8 always reuses the target head and has no clone override.
- **Depth three:** three alternating 1,024-token pairs improved the short
  fixture 14.8% and slowed 6K 21.9%, with changed transcripts. DFlash now uses
  the checkpoint proposal width (seven here); `mtpDepth` is a native-MTP
  control and cannot recreate this DFlash experiment.
- **Imported Splash packed Q4:** historical medians improved 32K decode 17.4%
  and reduced RSS from 21.01–21.09 GB to 18.40–18.50 GB, while short/6K decode
  regressed 8.4% / 25.6%. The metadata-selected packed loader and importer/tests
  were removed. The separate dense Q4 option and private head clone were also
  removed; that option requantized floating weights and never reproduced
  Splash's stored Q4 values or quantized selector.

Evidence: [head reuse pairs](../../../.cache/benchmarks/splash-qwen38-phase2/head-screen-summary.json),
[long head comparison](../../../.cache/benchmarks/splash-qwen38-phase2/long-head-summary.json),
[depth pairs](../../../.cache/benchmarks/splash-qwen38-phase2/depth-long-summary.json),
[follow-up interpretation](follow-up.md), and [historical exact Q4 comparison](exact-draft.md).

## Optimized paths already enabled

Manual rollback switches for the following paths have been removed. Shape,
dtype, device and safety eligibility checks retain necessary fallbacks:

- Segmented verifier attention and vector verifier attention splitting.
- Compiled verification where growing-prefix semantics are safe. Upstream
  `MLX_DISABLE_COMPILE` remains a shared MLX compiler control.
- Merged gate/up, merged K/V, stacked GDN input projection and reordered
  query/gate layout.
- Fused GDN preparation and window convolution.
- Fused draft convolution, top-16 and eligible greedy predecessor selection.
- Aliased immutable GDN snapshots and asynchronous prefill chunk submission.

Eligible two-column GDN recurrence and D256 full prefill attention also retain
their existing defaults with their rollback switches removed. Device and shape
eligibility remain; FP32 D256 attention still requires TF32. Removing those
manual overrides is not a new performance promotion.

The revised segmented attention change has the strongest isolated whole-model
evidence here: reversed-order pairs retained exact outputs/acceptance while
improving 6K decode 3.8%/3.1% and 32K 10.8%/6.0%. Short results were noisy.
The fused greedy walk reduced diagnostic selector time 899→238 microseconds;
the whole-model median difference was only 0.53% and does not establish a broad
throughput gain. These are retained historical measurements, not new claims
about the cleanup binary.

See [attention source](../../../crates/mlx-core/src/models/qwen3_5/attention.rs),
[compiled verification](../../../crates/mlx-core/src/models/qwen3_5/model/forward.rs),
[snapshot source](../../../crates/mlx-core/src/models/qwen3_5/layer_cache.rs),
[SDPA routing](../../../crates/mlx-sys/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp),
[segmented comparison](architecture-implementation.md), and
[selector comparison](../../../.cache/benchmarks/splash-qwen38-20260921/selector-ablation-summary.json).

## API defaults and controls outside this workload

The direct native chat API defaults `enableMtp` to false. `ChatSession` turns
it on automatically when a loaded Qwen model reports speculative weights,
unless explicitly overridden. The benchmark calls the native API with
`enableMtp: true`, a loaded draft, depth seven and adaptive depth off, and
asserts positive DFlash cycle counts. Therefore missing speculative enablement
does not explain its measured gap. Sources:
[native params](../../../crates/mlx-core/src/engine/params.rs),
[session configuration](../../../packages/lm/src/chat-session.ts),
[Qwen availability getter](../../../crates/mlx-core/src/models/qwen3_5/model.rs), and
[benchmark configuration](benchmark.ts).

Qwen paged early submission (`MLX_QWEN35_DECODE_EARLY_EVAL_LAYERS`, default
four) and unified-memory paged capture/restore (`MLX_PAGED_CPU_TRANSFER`,
default enabled) are also already active for their eligible callers. They do
not accelerate this DFlash flat-cache lane. Paged write/gather/grouped-attention
switches and the native-MTP expected-value controller likewise do not unlock
a missing DFlash path. See [execution plan](../../../crates/mlx-core/src/models/qwen3_5/model/chat_backend.rs),
[paged submission](../../../crates/mlx-core/src/models/qwen3_5/paged_forward.rs),
[paged storage](../../../crates/mlx-paged-attn/src/layer_kv_pool.rs), and
[native-MTP policy](../../../crates/mlx-core/src/engine/mtp_turn.rs).

`MLX_QMM_SPLITK_MIN_M`, `MLX_SDPA_BLOCKS`, `MLX_MAX_OPS_PER_BUFFER` and
`MLX_MAX_MB_PER_BUFFER` override shared backend heuristics. Forcing the tested
small-matrix route or larger command buffers did not establish a win. Their
existence is not evidence that a faster setting remains unused. Other-family
controls, including `MLX_QWEN4_*`, LFM2, Bonsai, Gemma and INT8-specific kernels,
were not performance-validated or recommended for removal by this audit.

Timing/tracing controls (`MLX_DFLASH2_PHASE_TIME`, `MLX_DFLASH2_VERIFY_LAYERS`,
`MLX_METAL_COMMAND_TRACE`, `MLX_METAL_OP_TRACE`) are diagnostics, not optimizations. The former
`MLX_REQUIRE_SEGMENTED_VERIFY_SDPA` test switch was replaced with a dedicated
strict test entry; production no longer reads it.

Runtime capability checks establish legal execution;
they do not automatically find the fastest kernel. Shared MLX routing still
includes architecture-class and workload thresholds. No new device-specific
constant, override or automatic performance claim is introduced by this audit.
