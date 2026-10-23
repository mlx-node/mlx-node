# Qwen MoE execution ports

The target is the 40-layer Ornith 1.5 35B-A3B GGUF: hidden width 2048,
256 routed experts, top-8 routing, intermediate width 512, 30 GDN layers,
and 10 full-attention layers. The official eight-row DFlash2 companion uses
seven proposal positions. Source reference: Splash revision
`1482f6bbc2abf12375332725302019594150b6cb`.

## Execution and cache ownership

MoE DFlash2 uses the whole-turn exclusive engine lane and flat target caches.
Native MTP capability remains separate; a draft companion does not advertise
native MTP weights or enter its continuous-batching lane. Disabling speculative
decode returns to ordinary paged/flat execution through the existing lifecycle.
Owner and conversation-root changes invalidate retained draft context. A warm
continuation reuses draft context only when its token provenance and target
frontier agree.

The verifier captures target hidden taps and GDN rollback tape. Compiled graphs
carry every layer's target state, RoPE base, optional destination blobs, and
optional live-row budget as inputs. Keys distinguish model identity, physical
verify width, KV format, destination mode, and budget mode. Prefix lengths may
vary only when attention can trace them safely. Unsupported geometry uses eager
verification. A compiled failure after store construction propagates and resets
the turn; retrying eager on potentially mutated caches is prohibited.

All physical verify rows are written before acceptance. A partial commit rewinds
full-attention offsets, replays only the retained GDN prefix, and appends only
retained hidden taps to the draft context. Full acceptance can swap packed GDN
state blobs. Stop, cancellation, and repetition limits reduce retention before
the single commit. Draft execution is submitted before building the target
verifier, and committed state/context stores are submitted at the cycle boundary.

## Projection fusion

`mlx_mixed_projection.cpp` combines two or three heterogeneous Tiled64
Q4_K/Q5_K/Q6_K projections at physical M=8. Each segment retains its original
split-K count and output allocation. This covers target GDN qkv/z and
full-attention q/gate, k, and v, including a previously merged k/v segment.
Contract misses retain individual projections. Tests compare raw BF16 output
bits for exact target dimensions, mixed quantization, and shared dense attention
view splitting.

The small split GDN b/a pair has a separate bounded optimization. Row-major
Q4_K/Q5_K/Q6_K with N=32+32 and K=2048 can share a 64-row projection at M=8;
the row-major `qmv_sg8` tier remains unchanged and its input preparation is
shared. Individual projection views remain authoritative for other widths.
Unrestricted merging is unsafe for bitwise parity because long-prefill
`qmm_splitk` partition counts depend on output width. The merged buffer replaces
the originals with row views, so retained weight storage does not increase.

## Adaptive verification

`mtpAdaptiveDepth: true` selects the model-specific GPU policy on unpenalized
greedy full seven-proposal blocks. `false` preserves fixed verification.
Sampled, penalized, teacher-forced, and shorter tail blocks retain complete
verification. Physical verifier width stays eight; the live-row scalar is a
lazy GPU dependency, not a host truncation or pre-verifier readback.

The selector produces confidence along the same predecessor path without
changing selected ids: `q = 1 / sum(exp(score - best))`. The B1 specialization
of Splash's `draft_verify_rows` maximizes expected retained tokens divided by
predicted cycle cost over live prefix lengths 1 through 8. Routed expert work
can be omitted for the suffix; every masked output must be finite zero. Other
attention/GDN computation, cache writes, and tape shapes remain physical width 8.

For live rows R, target row R-1 can label one extra draft proposal, while retained
cache rows cannot exceed R. Acceptance therefore computes raw matching labels
only through valid rows, caps accepted drafts to R-1, and uses target argmax at
the capped position as the boundary. Rollback still receives total_written=8.
Existing stop clamping can further shorten retention. Policy training uses raw
matches before that cap, never the stop-clamped count or an invalid suffix row.

The CPU policy is a B1 port of Splash `VerifyBudget.cpp`: prior slope 0.7,
position intercepts, 4096-label ring, 64-cycle IRLS refitting, 32-cycle full-row
exploration, and decayed linear cycle-cost fitting. A standalone reference
compiled from Splash and the Rust implementation matched every coefficient,
cost-table float bit, and ordered label at nine checkpoints through 3000
observations, including ring wrap. The checked-in golden unit test retains that
evidence. Policy state is per loaded model; successful committed cycles train
it after boundary submission. Adaptive cycles synchronize the committed graph
before taking the cost sample, so rollback and draft-context work are included
and host-only emission is excluded. Fixed cycles retain asynchronous boundary
submission without this added synchronization.

## Diagnostic controls

Set controls before loading a model or tracing a compiled verifier; changing
kernel toggles after a graph has been cached is not an A/B test.

| Control                             | Effect                                                                                                          |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `MLX_MOE_DFLASH2_COMPILE=0`         | Eager target verifier                                                                                           |
| `MLX_DFLASH2_COMPILE_DRAFT=1`       | Opt-in compiled full-width greedy proposal on GPU, including draft layers, target head and selector             |
| `MLX_DFLASH2_GDN_BLOB=0`            | Per-layer recurrent commit fallback                                                                             |
| `MLX_MIXED_PROJECTION=0`            | Individual heterogeneous projections                                                                            |
| `MLX_GDN_BA_FUSION=0`               | Individual small b/a projections                                                                                |
| `MLX_GDN_SCAN8=1`                   | Opt-in eight-SIMD-group recurrence schedule for complete D128/T8 GDN; other widths retain the existing schedule |
| `MLX_DFLASH2_SELECTOR_QUANT=1`      | Diagnostic Splash-style affine Q4/group64 selector hidden projection; default remains published BF16            |
| `MLX_DFLASH2_SELECTOR_EDGES_FP32=1` | Opt-in fused FP32 selector edge scores for rank 256/top-16; default retains native BF16 intermediate arithmetic |
| `MLX_KQUANT_REGISTER_OUTPUT=1`      | Opt-in T64 register MMA for M8/Q4K target mixer output dimensions; separate numerical experiment                |
| `MLX_KQUANT_REGISTER_AFFINE=1`      | Opt-in register MMA for selected M8 affine Q4/group64 draft projections                                         |
| `MLX_KQUANT_STAGED_SPREAD=1`        | Opt-in staged M8 K-loop spreading on Apple GPU family 10 or later with at least 16 cores                        |

The selector quantization option can change proposals/acceptance and is a
separate numerical experiment. The exact companion source alone does not imply
identical selector arithmetic across engines.

The FP32 edge experiment follows Splash's eight-successor SIMD reduction and
keeps predecessor-times-hidden products and edge sums in FP32. It feeds those
scores directly into the existing path/confidence kernel. Candidate order and
the native tie rule remain unchanged, so this is not a claim of complete source
selector bit parity. It is independent of the selector projection's Q4 option.

The register experiment preserves row-major SG8 bits after T64 repacking. On
its deterministic M8/N2048/K4096 fixture, independent F64 accumulation over
CPU-dequantized weights measured 0.2263% peak-normalized error and 0.1664%
relative RMS, compared with 0.2306% and 0.1671% for the existing half-NAX path.
This establishes bounded operator accuracy, not full-model quality or speed.

The staged-spread experiment follows Splash's column offset
`begin + (((tile * 40503) & 65535) * (end - begin) >> 16)` and wraps inside each
K split. Split counts, half weight conversion, and split reduction order stay
unchanged, but the within-split accumulation order changes. It applies to
ordinary physical-M8 staged projections and each fused mixer's local column
tile; grouped experts remain lockstep. Full-model teacher checks and isolated
on/off timing are required before enabling either experiment by default.

## Remaining performance work

The compiled-draft experiment keeps RoPE positions, selector anchor ids, sliding
masks, full K/V owners and live-window views as explicit inputs. Its in-place
tail store resolves physical offsets and lengths at replay, checks every alias
before recording writes, and returns both full buffers and attention views so
the next accepted-context append stays ordered. The first full-width greedy
proposal uses the eager lane and materializes captured weights. Sampled,
penalized, CPU, and short final proposals retain eager execution. This avoids
compiling a new graph for a tail width used only once. A model-owned namespace is
erased on drop or target
weight replacement; an invoke error resets the turn instead of retrying after
possible partial stores. The independent-storage replay fixture checks exact
paths, confidence, and every K/V byte while positions, anchors, view offsets,
capacities, and mask mode change. Replacing the target embedding and invalidating
the namespace triggers a fresh trace. Unrelated views fail before either cache
is written. These gates passed, along with the existing dense residual/mask
regression. Frozen V8 passed all nine live acceptance cases with compiled draft
enabled for each of BF16 and INT8 target KV: warm continuation, sync/stream text,
cancellation and recovery, sampled penalties, and AR-to-DFlash transitions.
The experiment remains opt-in while full-model timing is evaluated.

The long-context window has the same token visibility in both engines. Splash's
2048-slot ring reads at most 2047 old rows; native retains 2047. At proposal row
`i`, both mask old positions earlier than `query_position - 2047`, and both expose
all eight current proposal rows. They use absolute positions and the companion's
default RoPE base of 10,000,000. Frequency construction, trigonometric kernels,
attention reductions, and selector arithmetic can still change numeric results.

A native phase-synchronized v1 profile measured approximately 3.60 ms for draft
and selector and 18.64 ms for target verification. Splash's 2K fused cycle was
14.42 ms GPU / 15.04 ms wall; its isolated per-dispatch replay totals are larger
and cannot be added as fused-cycle costs. These are historical diagnostic
measurements, not final performance claims.

Deferred GDN commit inside the next verifier could remove redundant recurrent
state traffic, but the entire native post-commit boundary measured only about
0.45 ms/cycle in that profile. It needs careful stop/warm-reuse materialization
and should follow stronger evidence. Shared command submission remains an
architectural difference. Increasing global MLX command
buffer thresholds was measured slower and is not enabled. Command-buffer and
stream tags allow primitive records to join completed commands in completion
order, with addresses treated as reusable. Mixed command intervals still do
not provide per-kernel timings.

The V10 frozen gates passed for the eight-SIMD GDN schedule (all eight outputs
and destination aliases exactly match, including one compiled graph replayed
at T=8/7/1/16/8), FP32 selector edges (independent F64 scores/confidence,
changed anchors/codebooks, invalid contracts), the production BF16-unary
widening hook, and the compiled draft/shared dense regressions. Logs are
`v10-execution-gates.log` in the research artifact directory. The isolated
GDN ABBA medians were 30.32/27.75/28.49/29.82 microseconds (off/on/on/off);
these are operator observations during the canonical CPU build, not a product
throughput claim. Exact-ID model traces also matched across the GDN toggle.

`MLX_DFLASH2_ATTENTION_NAX=1` enables the new source-style draft attention
experiment only for BF16 B1/Hq32/Hkv8/T8/D128 and live KV lengths 8..2055 on
Metal tensor-op hardware. It uses four splits and FP32 online softmax/partials
(~0.51 MiB per layer), while the existing MLX vector path uses 128 splits at
the long-window shape. The helper consumes the current explicit boolean
mask and KV views, including head-capacity gaps and nonzero offsets; replay
recomputes length and strides. Dynamic tensor slices bound partial context
tiles to initialized rows. Eight current proposal rows use the source's
16-row threadgroup padding for the value product. Contiguous live-window
tile order differs from Splash's physical-ring tile order, so numerical
oracle and exact-transcript gates are required before enabling it by default.
The default and unsupported shapes/devices retain generic attention. The
raw helper defines a fully masked row as zero. No KV storage is mutated by
attention. The initial V12 raw gates caught an incorrect dynamic value-tensor layout:
current-only windows passed, but partial old-context tiles read guard NaNs.
The corrected kernel constructs bounded 64-wide chunks explicitly and uses
native token-major V with an untransposed MPP operand, matching the proven
segmented-attention layout. The corrected Metal-only iteration passes 16 F64
oracle cases and seven changing-length/stride/offset/mask replays through one
compiled graph. Across those cases, worst maximum error / oracle peak is
0.003464 and worst RMS / peak is 0.000526; single-visible-key outputs match
exactly. Its passing log is `v12-attention-rowmajor-v.log`; the failed as-built
log remains `v12-attention-as-built-gates.log`. The shared production hook
accepts real fused QK norm/RoPE outputs and matches the raw helper plus output
projection exactly. Model route, transcript, live lifecycle and product
performance gates remain required. Base and row alignment is conservatively
checked; an unsupported lazy view discovered only at evaluation fails closed.
