# Grouped GGUF MoE port

The port follows Splash commit `1482f6bbc2abf12375332725302019594150b6cb`,
especially `runtime/ops/MoE.cpp`, `shared/moe_gguf.metal`,
`common/gguf_column_tile.h`, and the staged tensor-op helpers. Its Apache-2.0
attribution is in `THIRD_PARTY_NOTICES`.

`mlx_splash_moe.cpp` owns a lazy MLX primitive with device-side routing,
expert grouping, activation gather, fused gate/up, down projection and weighted
combine. The Metal kernels accept the project's lossless Q4_K, Q5_K and Q6_K
three-plane storage, including its Tiled64 layout. Each of the six routed/shared
projections can use a different format. No FP16/BF16 conversion of stored
quantized weights is performed. Loading tiles the routed planes only when the
complete fast path is eligible, and transfers their ownership out of the source
weight map instead of retaining a second resident copy.

The eligibility boundary is GPU tensor-op support, BF16 activations, 1–2048
physical rows, at most 256 experts/top-8 routing, and hidden/intermediate widths
divisible by 256. CPU selection, gradient tracing, unsupported quantization or
geometry, and `MLX_SPLASH_MOE=0` retain the generic path. Inference compilation
does not disable it. Replay validates weight and routing geometry; temporaries
are registered with the MLX command encoder to retain their lifetime.

One or two rows for an expert use a direct column dot in the native vector
precision tier. The physical BF16 grouped-GEMM tier uses MPP8 even for those
rare experts, preserving native accumulation order during prefill. Larger occupancies use
8/16/32-row MPP tiles, with a fused gate/up dispatch and independent simdgroup
stages. The maximum dispatch grid is bounded without reading routing back to
the CPU. `MLX_SPLASH_MOE_INDIRECT=1` instead dispatches exactly the live expert
tiles. Grouping writes both three-dimensional grids beside the tile count;
the two projections bind that buffer as an input and retain its storage with
their other workspaces. The MLX encoder extension preserves its normal hazard
barrier and dispatch accounting. This experiment requires the local MLX
`device.h` change as well as the matching bridge and Metal library.

An optional device int32 budget retains the physical output shape and the
physical-row precision policy. Only live routed rows enter route/group work;
shared expert tiles still cover the physical width, as in Splash. Combine
writes exact positive zero for dead rows before reading their uninitialized
routing slots. The budget is a graph input, so changing its value does not
require retracing or a host readback.

## Arithmetic contract and diagnostics

The imported checkpoint's routers and shared scalar gates are BF16. Native
softmax therefore has BF16 intermediate boundaries; top-k operates on those
rounded probabilities, with stable ascending ordering and larger IDs winning
ties. The top-8 denominator and expert sum also round after **each BF16 add**.
The grouped implementation preserves these boundaries, the native stable
sigmoid expression, BF16 projection outputs and BF16 SwiGLU intermediates.
Routing-only tests cover ties, unequal probabilities, maximum geometry and
nonfinite inputs, including bounds-preserving expert IDs.

Projection weight precision follows the existing native dispatch tiers rather
than assigning one precision to every workload. Native grouped GEMM stages
BF16 decoded weights; the shared M8 Tiled64 tier stages half; vector gather
decodes F32. The uncommon dense expert tile in the F32 tier uses high/low half
matrix passes to retain the decoded weights' precision.

The native vector kernel is not an exact decoded-weight oracle. Its affine
bias correction adds several BF16 input values before promoting their sum to
F32. A direct dot of independently decoded weights avoids that extra rounding.
On a sampled real layer-0 M8 expert, grouped gate/up/SwiGLU were bitwise equal
to independent decoded-weight F64 accumulation; down was 99.90% equal with
relative RMS error 0.000050. Native down had relative RMS error 0.010878 to
the same oracle. The diagnostic reports native differences separately and
asserts the independent oracle for that tier. It does not loosen the established
synthetic regression thresholds to accommodate native vector rounding.

The initial real-weight random-input probes were almost bitwise exact in the
M2047 tier. Capturing all 40 layers' actual teacher inputs exposed the remaining
seed differences: rare one/two-row experts still used the column accumulation
order while native used grouped GEMM. Before correction, the worst combined
relative RMS error in one layer was 0.00001311, with at most 36 differing
elements among 4,192,256. Such tiny differences can affect later discrete
routing decisions. Restricting the column path to the native vector tier made
combined output bitwise equal for **all 40 captured real layers**, with exact
route IDs and coefficients throughout. The subsequent full-model comparison
over 2047 prediction positions was also exact: top-1 agreement 1, KL 0 and
identical NLL 1.4252860731302848. The raw comparison is retained with the
`candidate-v6-prefill-order` benchmark artifacts.

`tests/splash_moe.rs` includes an ignored bounded real-layer diagnostic. Set
`SPLASH_MOE_WEIGHTS` to the prepared `model.safetensors`, optionally
`SPLASH_MOE_LAYER`, `SPLASH_MOE_ROWS` or `SPLASH_MOE_ROUTE_PATTERN=skew`.
`SPLASH_MOE_INPUT` consumes a saved tensor file containing `x`, `router_logits`
and `shared_logits`; it overrides synthetic input generation and row count.
The saved 2047-row teacher inputs remain available in the research artifacts;
the temporary model capture hook has been removed.

The focused suites are `splash_moe`, `splash_moe_budget` and
`mixed_projection`. Budget coverage includes row-major/Tiled64 planes, mixed
formats, BF16/F32 router outputs, repeated scalar changes through one compiled
graph, exact dead-row zeros, full-budget equality and live-prefix comparisons
when expert occupancy crosses from column to matrix execution.

## Decode experiments

The ignored actual-layer test also accepts `SPLASH_MOE_BENCH=1` for a bounded
M8 operator diagnostic. It builds 16 independent lazy roots before each timed
evaluation, reports graph construction separately, and retains resident inputs
and weights. `SPLASH_MOE_PROFILE=1` measures successive route/group/gather/
gate-up/down/combine dispatch prefixes. These are diagnostic timings, not
compiled whole-model throughput. `MLX_METAL_COMMAND_TRACE` can additionally
report GPU command intervals and barrier/resource counts. The optional
`SPLASH_MOE_BUDGET` fixes a device row budget while keeping physical M8;
the printed occupancy histogram distinguishes column and staged work.

Vector activation loads, retaining staged up results in registers, and extending
the routed column threshold to four rows have not shown a repeatable operator
improvement and are not enabled. An out-of-line column helper also passed the
numerical gates without a measurable gain. Source-like half staging for routed M8 matrix
tiles is retained only as an isolated experiment. It preserves the F32 column
path and BF16 prefill tier but changes decoded-weight rounding for staged
decode experts. With actual layer-0 weights and a concentrated seven-row-per-
expert fixture, it reduced grouped operator time from 120.88/122.77 us to
86.46 us while the unchanged native controls stayed at 213.76/213.41 us.
Independent F64 oracle errors for the staged sample were about 0.10% RMS for
gate/up and 0.215% for down, within the existing numerical gates. The captured
layer-0 input with mostly one/two-row experts showed no clear gain
(218.33 versus 219.41 us). Real generated route distributions and full-model
quality/acceptance must establish whether this tradeoff is useful.

The V8 profile used five saved layers (0/5/10/20/39), each with 34–52 active
routed experts and 3–7 experts requiring staged execution. Route, group and
gather together took approximately 23 us; gate/up added 84–109 us and down
49–81 us. Thus projection work dominates this fixture, while gather contributes
only a few percent. Half staging reduced the five complete grouped medians by
2.7–9.2%, with explicitly sampled four/five-row experts passing the independent
oracle (worst sampled down RMS error 0.233%). The raw logs and summary are in
`v8-grouped-profile`; these sequential operator timings have no clock telemetry
and are not a whole-model speed claim.
