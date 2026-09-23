# Bounded DFlash2 state and workspace follow-up

**Historical research notes.** The subsequent experimental compiled-settlement
path and its switch were removed; see [the cleanup decision](cleanup.md).

Read-only source audit, September 22, 2026. This note covers the dirty optimized
mlx-node checkout and `/Users/brooklyn/workspace/github/splash`. No runtime
edits, builds, model loads, or GPU tests were performed. Numbers below count
tensor payloads and source-level allocation requests or loads/stores, **not
physical DRAM traffic, measured peak residency, or predicted speedups**.

## Most useful new bounded change: omit dead verifier state outputs

The compiled DFlash2 verifier needs each GDN layer's output activation and replay
inputs, but does not need its final recurrent or convolution state. The current
graph drops those state handles, yet their multi-output custom kernels still
allocate and write the state outputs. Add an explicitly verifier-only output
policy to those primitives before attempting a persistent whole-model arena.

Evidence in the current checkout:

- `crates/mlx-core/src/models/qwen3_5/model/forward.rs:643-657` creates a detached
  linear cache from graph inputs, forwards the layer, and exports only
  `q,k,v,g,beta,qkv`; its comment explicitly says verify state writes are dropped.
  The output contract at `:507-515` excludes final GDN state. Ordinary attention
  new K/V remain outputs and must not be dropped.
- `crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs:564-575` stores the fused
  prepare kernel's history only in that cache. At `:661-682`, recurrent state is
  only assigned into the cache; downstream normalization and projection use `y`.
  Neither final state is an input of the next decoder layer.
- `crates/mlx-sys/src/mlx_gated_delta.cpp:92-96,207-220` declares and returns both
  `y` and `state_out`. The default two-value-column shader's last loop writes
  final state (`src/metal/common/gated_delta_step_2vcol.metal.inc:93-96`) after
  all activation outputs have been computed. The one-column and opt-in
  four-column variants likewise have terminal stores at
  `gated_delta_step.metal.inc:65-68` and
  `gated_delta_step_4vcol.metal.inc:110-116`.
- `crates/mlx-sys/src/mlx_qwen4_gdn.cpp:140-155` declares six prepare outputs,
  including `[3,10240]` BF16 `next_history`. Its only stores are a distinct
  `token == 0` block in `src/metal/qwen4/gdn_prepare.metal.inc:25-30`; current
  convolution values read old history and raw QKV independently at `:15-23`.
- MLX does not eliminate one unused custom-kernel sibling: `mlx/mlx/compile.cpp`
  under `crates/mlx-sys` keeps a primitive unless **all** its siblings are dead
  (`:768-775`) and recreates all outputs during replay (`:1063-1080`).
  `mlx/backend/metal/eval.cpp:75-85` passes `arr.outputs()` to the evaluator;
  `mlx/backend/metal/custom_kernel.cpp:32-38` allocates every output.
  `mlx/backend/metal/device.cpp:453-466` also retains sibling storage until
  command completion. Dropping the Rust handle is therefore insufficient.

For the supplied target geometry, 48 GDN layers have recurrent state
`[1,48,128,128]` in BF16 for this runtime lane, and convolution history
`[1,3,10240]`. Removing these **verifier-only** stores eliminates the following
source work per cycle, independent of retained count:

| Output omitted                           | Payload per layer | Payload across 48 layers | Allocation requests removed |
| ---------------------------------------- | ----------------: | -----------------------: | --------------------------: |
| Final verifier recurrence                |       1,572,864 B |                   72 MiB |                          48 |
| Final fused-prepare history              |          61,440 B |               2.8125 MiB |                          48 |
| Combined, when fused prepare is eligible |                   |              74.8125 MiB |                          96 |

These are 96 calls to the MLX allocator, not necessarily 96 new Metal buffers:
the allocator can reuse cached buffers. The 72 MiB source state read by verify,
its recurrence arithmetic, activation writes, replay input tape, and the later
72 MiB replay read plus 72 MiB committed-state write remain. The proposal is
not a way to eliminate the mandatory serial-equivalent replay. The target JSON
contains `mamba_ssm_dtype: float32`, but this lane's actual primitive output
dtype is `q_arr.dtype()` (`mlx_gated_delta.cpp:152-154,209-210`), and replay uses
the recorded key dtype (`:530-566`); config text alone must not determine the
byte accounting.

Prototype scope:

1. Introduce an explicit `verify_outputs_only` policy at the **compiled detached
   verifier** callsite, not a global inference flag and not merely
   `record_tape=true`. Other tape consumers can require live state.
2. Preserve exactly the selected arithmetic source, reduction order, SIMD lane
   map, FP32 carry, BF16 output casts, and compile options. Generate a matching
   one-output recurrence signature with only the terminal state pointer/stores
   omitted. Kernel cache identity must include the output policy as well as
   mask/vector/column policy; compiled graph identity must not reuse a trace
   built under the opposite policy.
3. Preserve current default two-column selection
   (`mlx_gated_delta.cpp:174-185`). Support legacy one-column when explicitly
   selected and opt-in four-column when explicitly selected; do not enable the
   four-column policy as part of this change. The vector four-column FP32 path
   is gated to another workload and `T > 8` (`:186-200`) and is out of scope.
4. For the current eligible fused prepare path, create a five-output signature
   omitting only history. Keep mean-epsilon and BF16-beta settings passed at
   `gated_delta_net.rs:535-548` and all Q/K/V/gate computations. When prepare is
   disabled, unsupported, or declines, retain the existing prepare path first;
   claim only the 72 MiB recurrence saving there. Fallback AR, prefill, masked
   recurrence and other model families retain their state-producing policy.
5. No persistent mutable storage is needed for this first experiment. On any
   unsupported shape or pipeline, use the established two-output implementation.

Source separation establishes feasibility, not measured exactness: changing
shader outputs can alter compiler allocation/code generation. Require bit-exact
`y`, tape, target logits and committed continuation tests before accepting it.

## What already exists, and what must not be counted again

The earlier experiments are recorded in `architecture-implementation.md`.
Segmented target verifier attention is already enabled. Cache reservation,
earlier settlement, direct convolution replay and retaining more allocator cache
did not establish worthwhile standalone end-to-end gains. Do not label them new
work here. `follow-up.md:40` records a previous full draft ring experiment that
was exact in its kernel tests but 10.5% slower in append-plus-attention screening;
repeating that replacement unchanged is not justified.

GDN snapshots already alias immutable state descriptors by default:
`layer_cache.rs:149-167`; the old defensive copies are opt-in. The earlier
snapshot API documentation at `:92-118` is stale relative to that implementation.
There is no remaining 72 MiB snapshot-copy saving to claim. The default
compiled verifier already exists and returns its replay tape in one invocation
(`model/forward.rs:497-528,700-709`).

## Replay tape size and lifetime

Metadata was read without weights: `.cache/models/qwen3.8-27b-gguf/config.json`
has 64 layers, one full-attention layer per four, Hk=16, Hv=48, Dk=Dv=128 and
convolution width four (`:89-101`, layer list above). The draft config has five
layers, eight KV heads, head dimension 128, five 5,120-wide target taps and a
2,048-token window (`.cache/models/qwen3.8-27b-dflash2/config.json:14-22,24-51`).

`GdnKernelTape` records exact Q/K/V/g/beta inputs as lazy clones
(`gated_delta.rs:299-325,966-978`); `GdnLayerTape` additionally holds raw QKV
(`gated_delta_net.rs:25-28,482-484`). GGUF keeps compact tiled key heads rather
than repeating them to Hv (`gated_delta.rs:894-916`). Current eligible prepare
emits BF16 Q/K/V/beta and FP32 decay (`mlx_qwen4_gdn.cpp:146-155`). At batch one
and verify width eight, logical payload sums are:

| Tape payload across 48 GDN layers      |      Bytes |         MiB |
| -------------------------------------- | ---------: | ----------: |
| Q + K + V                              |  7,864,320 |         7.5 |
| Decay + beta                           |    110,592 |  0.10546875 |
| Raw convolution QKV                    |  7,864,320 |         7.5 |
| Total current tape                     | 15,839,232 | 15.10546875 |
| Total omitting post-verify Q retention | 14,266,368 | 13.60546875 |

This is a payload sum, not a bound on current backing allocation/graph residency:
views can retain larger projections, custom inputs can get contiguous copies
(`custom_kernel.cpp:41-48`), and command completion retains graph inputs. Actual
unique backing capacities must be measured separately before claiming a peak
memory reduction. Avoid packing the current lazy tape into fresh buffers as a
standalone optimization: recording presently adds no explicit copy.

Fused replay never consumes Q (`gated_delta.rs:352-359`;
`metal/common/gated_delta_replay.metal.inc:1-7`). Q remains needed by the existing
fallback T=1 kernel, whose unused activation uses Q (`gated_delta.rs:362-375`).
A state-only fallback could reuse K as a shape-compatible dummy query because
recurrence is independent of query, or expose a true state-only primitive. This
can shorten Q lifetime after verify; it cannot remove producing Q for verify.
Keep this small 1.5 MiB lifetime change separate from the dead-output experiment,
and prove fallback exactness before changing the tape ABI.

An eight-prefix recurrent-state tape would write 8 x 72 = 576 MiB per cycle and
requires computing the BF16-rounded recurrence in addition to verification's
FP32 carry. A direct final-state selection is numerically invalid. Current
fused replay rounds after every token (`gated_delta_replay.metal.inc:45-74`) and
must remain the state oracle even at full acceptance
(`dflash2_decode.rs:422-437`).

## Persistent workspace: small ownership boundary, not one giant arena

A second, larger prototype can own two reusable per-layer GDN state banks and
fixed width-eight tape/commit scratch for one exclusive DFlash2 session. The
starting implementation should reuse buffers only for commit outputs, retaining
current verifier arithmetic. Existing custom kernels always allocate outputs,
so true output binding requires a dedicated primitive/FFI owner; ordinary
`fast::metal_kernel` plus a preallocated input array does not accomplish it.
Buffers supplied as inputs must never be silently written by an allegedly pure
kernel.

Define an epoch-owned transaction: immutable `current`, private `candidate`,
verify tape/taps/new target KV, accepted IDs, host-clamped `keep`, and completion
token. Verify reads only `current`; replay writes `candidate`; successful
completion atomically publishes parity, target/draft frontiers, provenance and
next anchor. Never overwrite either bank while a lazy array, pending command,
snapshot or persistence reader leases that epoch. Initial scope should allow
one in-flight cycle and one exclusive owner. Admit a replacement bank or export
an owned immutable snapshot when a continuation/prefix cache retains a bank;
otherwise fall back. Never silently create a third unbounded bank.

Two recurrent-plus-convolution banks are 149.625 MiB of logical payload. With
one current 15.1055 MiB tape, a 39.9805 MiB draft tail, 0.390625 MiB of taps,
0.5 MiB of width-eight target K/V and 0.15625 MiB draft K/V staging, this selected
state/scratch set is approximately 205.76 MiB per exclusive owner, **excluding**
weights, full target KV, logits, projection/attention scratch, MLX graph objects,
allocator alignment/cache and command retention. Target BF16 KV grows by
65,536 B per context token (16 layers x four heads x 256 x two bytes x K/V),
about 2 GiB at 32,768 tokens. Derive the complete admission budget using checked
arithmetic and actual unique buffers, runtime allocator/device headroom and
max buffer length; no model-name-specific device rule. An eight-state tape is
explicitly outside this budget.

Two banks remove repeated allocation requests for replay destinations, but do
not remove recurrence read/write bytes. Unlike the dead-output proposal, this
is mainly allocation/ownership work; the allocator already caches freed buffers
(`mlx/backend/metal/allocator.cpp:123-175,183-200`). Its global mutex/cache lookup,
resource limits and periodic clears are real source costs, but no timing gain
is established. A bounded active workspace survives `clear_cache` because it
remains live; this does not justify disabling the periodic synchronization or
memory guard (`engine/dspark_turn.rs:1211-1217,1408`).

Keep distinct backing buffers for independently scheduled categories and parity
banks. MLX's hazard keys are raw `MTL::Resource*`, ignoring subrange offsets
(`device.cpp:409-445`); conflicts produce buffer-scope barriers
(`:476-499`), and cross-encoder dependency maps are likewise buffer-keyed
(`:549-562`). Putting independent layer tapes, all activations and current/next
state into a single MTLBuffer would introduce false dependencies between
nonoverlapping views. A scoped primitive can perform several internally ordered
dispatches, but must correctly register every outer input/output and retain the
owner. Replacing the general MLX tracker with range-aware hazards is a separate
backend project, not a prerequisite for the bounded prototype.

## Draft context, shared prefixes, and transaction safety

Draft context append projects only accepted tapped rows today
(`dflash2_decode.rs:193-207`), then per-layer K/V (`dflash2.rs:225-239,735-774`).
Its tail limit is 2,047, not 2,048 (`dflash2.rs:725-728`): proposal rows complete
the attention window. Multi-row append routes to `RotatingKVCache::update_concat`
(`transformer/rotating_kv_cache.rs:373-379`), builds old-tail-plus-new K/V at
`:224-250`, and discards the returned full view in DFlash append. Proposal then
concatenates retained tail with its own block K/V (`dflash2.rs:261-277`).

For a full tail, one old-tail copy across all draft layers covers
5 x 2047 x 8 x 128 x 2 x 2 = 41,922,560 B, approximately 39.98 MiB of payload.
A multi-row append concatenation logically reads that old payload and writes it
into the new result: about 79.96 MiB of old-tail read+write accounting, plus new
rows. Proposal's separate concat has a similar old-tail payload. A ring with
direct chronological addressing could remove those materializations, but the
already rejected ring kernel demonstrates why that byte arithmetic is not a
performance result. The established attention reduction/order must be retained;
do not replace it with an unrelated custom reduction solely to avoid copies.

An independently useful experiment is a DFlash-only append API that does not
construct a full attention return value, uses bounded private tail storage, and
keeps existing proposal attention. It must also avoid reconstructing the entire
temporal order in `fetch_current_kv` (`rotating_kv_cache.rs:584-592`); otherwise
copies simply move to the consumer. With current attention requiring a dense
chronological tensor, total copy removal needs either a compatible segmented
view implementation or a different cache layout. Report the remaining copy
explicitly; do not rebrand ring allocation as copy elimination. Single-row
updates have donation-dependent copy behavior: `SliceUpdate` calls `copy_gpu`
first (`mlx/backend/metal/indexing.cpp:746-764`), which elides the base copy only
when donation is legal (`metal/copy.cpp:13-23`, `common/copy.h:30-46`).

Raw-accept device commit cannot write a wrapping draft ring in place before host
clamp. Rejected suffix rows can overwrite old history needed by the shorter
accepted window. Stage at most eight new K/V rows per draft layer (160 KiB total)
and scatter only after clamp, or use an undo log of every overwritten slot with
failure-safe restoration. The simpler first prototype is staged writes. Full
ring ping-pong doubles 39.98 MiB and still copies unchanged contents unless using
an overlay-aware attention consumer.

Publishing immutable/shared prefixes needs more than refcounted mutable arrays.
The current full target cache writes through its held handles
(`transformer/kv_cache.rs:112-122`); snapshots retain only offsets
(`qwen3_5/layer_cache.rs:145-147`). An external prefix consumer must receive an
owned snapshot, immutable chunk ownership with copy-on-write tails, or an
explicit lease preventing mutation. First prototype should exclude shared
mutable target prefixes. Flat/paged lane ownership and exact history remain
mandatory: tests at `dflash2_decode.rs:1200-1209` reject equal-length unrelated
history and paged-owned target state. A future immutable prefix/private suffix
design also needs attention support for the actual number of segments; the new
two-segment target verifier does not automatically implement arbitrary paging.

## Prepared device commit while preserving the current stop contract

The host clamp invokes observers once, checks cancellation before observer/EOS,
and excludes a tripping accepted token's cache slot while retaining its emitted
history (`engine/dspark_turn.rs:395-480`). `emit_count`, `keep`, boundary token
and anchor are distinct. Preserve those semantics, including cancellation that
arrives while the GPU is executing; do not silently adopt Splash's narrower
device-only stopping contract.

A safe staged route is:

1. Device acceptance prepares raw IDs/count and GDN replay into the inactive
   state bank, retaining old state, tape, taps and new target K/V. It does not
   publish or overwrite active draft history. Wait once for that prepared result.
2. Run the existing host clamp and observers exactly once. If the raw prepared
   count equals clamped `keep`, retain prepared GDN state. If shorter, rerun exact
   replay from **old current**, overwriting inactive state after its pending
   write completes. Preserve complete fallback roots until publication succeeds.
3. Commit target KV only for the clamped prefix; project and append draft context
   using the same retained row shape as baseline; complete/surface errors, then
   publish epoch/frontiers/history together. If new primitive eligibility or
   allocation fails before submission, use the existing path. After a partial
   submission, drain its private work before reusing storage or falling back.

This can move GDN settlement before CPU observation; it does not remove the
host decision boundary or all post-clamp work. Precomputing draft projections
for all eight rows and selecting a prefix is **not assumed bit-exact**: baseline
projects only `keep` rows, and one-row versus multi-row matmul policy can differ.
Retain eight shape-specialized projection plans or prove row-shape parity before
including draft projection in the prepared device graph. A device-count GDN
replay kernel has the same token loop; allocation/launch eligibility still must
inspect running pipeline limits. No chip-generation tuning is needed.

## What Splash actually owns and commits

Splash's `runtime/model/RuntimeArenas.hpp:314-346` allocates one shared decode
base with shape-derived views and separate private gate scratch. Its packed
layout reserves maximum lane width with no per-lane alignment holes
(`:308-310,359-406`); it is not a dynamically reusable MLX graph allocator.
`RuntimeArenas.mm:141-155,280-296` sizes per-layer replay inputs and chunk K/V.
`engine/MemoryPlan.cpp:94-105` counts shared prefill/decode and pipeline/overhead
reserves alongside weights before dynamic memory admission.

State lives separately: `QwenState.cpp:36-65` allocates a GDN cell and its views;
`:135-140` initializes active parity, and `Runtime.mm:1412-1435` binds current
and next state plus device retained count. Acceptance, recurrence commit and
draft commit are encoded together (`Runtime.mm:2253-2261`), and CPU finalization
swaps parity only after reading/validating results (`:1466-1508`). Draft commit
binds persistent rings and retained count (`:1349-1359`), while
`DFlashDraft.cpp:251-270` projects all fixed verify rows before its gated ring
write. That latter shape policy is not interchangeable with mlx-node's
keep-shaped BF16 projection without validation.

Splash's normal backend records prepared dispatches through one ordinary Metal
compute encoder (`runtime/metal/MetalBackend.mm:1414-1439`) and uses completion
tickets (`:1449-1451,1488-1497`). It does not use MLX's resource-set hazard
inference, so copying its arena layout into MLX does not reproduce its ordering.

Immutable Splash cache state is not zero-copy: `QwenState.cpp:196-224` acquires a
pooled cell and ring and `memcpy`s active GDN plus every draft K/V buffer.
Destruction returns storage to pools (`:89-94`); idle release is bounded
(`:163-170`); allocation is admitted (`:350-379`). Snapshot validation requires
complete draft window and page-aligned committed target length (`:408-420`).
These are important lifecycle costs/constraints, not proof that immutable
sharing is free.

## Exact tests and acceptance gate for later implementation

No tests were run in this research-only turn. A later implementation should use
the current optimized artifact as oracle, and retain these distinct gates:

1. **Dead-output shader gate:** widths 1 through 8, every supported existing
   column policy, prepare enabled/disabled, cold zero and nonzero state, compact
   GGUF key heads, BF16 gates/FP32 decay; bit-compare all `y` rows and Q/K/V/g/beta
   tape. Also exercise unsupported shape/mask/backend fallback. Check the emitted
   primitive really has one recurrence/five prepare outputs; count allocation
   requests and source output bytes separately from new Metal allocations.
2. **Every acceptance prefix:** for every verify width 1 through 8, test every
   legal keep 1 through width. Compare target K/V values/frontier, recurrent
   state, convolution state, all draft cache rows, draft logical offset,
   provenance, next anchor and next proposal. The reference is the unchanged
   verifier plus exact accepted-prefix replay, supplemented by the existing
   T=1 recurrence oracle (`gated_delta.rs:1197-1225`); immediate argmax parity
   alone is insufficient.
3. **Continuation:** several consecutive cycles, full/first/middle rejection,
   tail boundaries 2046/2047/2048/2055 and multiple wraps, cached second turn,
   cold replay of the same transcript, reset, explicit AR fallback and restore.
   Reuse the state-sensitive existing fallback fixture
   (`dflash2_decode.rs:990-1005,1213-1227`) and durability/provenance fixture
   (`:1231` onward). Validate actual target/draft rows after a keep=1 cycle
   followed by keep=8; this catches stale ring slots and bank reuse races.
4. **Stopping and failure:** EOS/extra EOS at every accepted position and at
   boundary; budget exhaustion, repetition and force-think-end; observer stop
   called exactly once; cancellation before, during and after verify and at
   each clamp position; blocked streaming callback; error at each allocation,
   submission and completion boundary. Prepared-state mismatch must replay from
   old current and preserve emitted-history/cache-frontier distinction.
5. **Ownership:** retain an old snapshot while cycling banks, drop a session
   during pending work, clear allocator cache with in-flight work, restore into
   another owner, equal-length wrong provenance, and shared-prefix mutation
   attempts. Failed admission must leave the old owner usable, or invalidate
   the session explicitly if device execution failed. Never publish half-updated
   target and draft state.
6. **Performance:** only after exactness, alternate isolated uninstrumented
   processes with identical target/draft, IDs, acceptance, output hashes,
   prompt/cache state and lengths. Report full cycle/turn latency and peak
   unique-buffer residency. Use command diagnostics separately to count removed
   output requests, temporary lifetimes, barriers and allocator misses; an
   allocation-byte reduction is not a measured DRAM-bandwidth reduction.

Priority order is dead verifier outputs, then explicit bounded commit ownership
if allocation/encoding evidence justifies it. A new ring implementation or fully
device-controlled session publication needs its own evidence and broader tests.
This ordering ranks feasibility and strength of source evidence, not expected
speedup. Removing 74.8125 MiB of logical stores is bounded cleanup alongside
multi-gigabyte weight reads and the remaining proposal/verify/replay work; it
does not establish an explanation or solution for the roughly twofold engine
gap. The larger execution-ownership redesign has broader potential but also
much greater implementation and correctness uncertainty.

## Subsequent authorized prototype: compiled post-clamp replay

After the research pass, a bounded implementation was added under
`crates/mlx-core/src/models/qwen3_5/dflash2_commit.rs`, wired only through the
DFlash2 stepper. The now-removed `MLX_DFLASH2_COMPILED_COMMIT=1` switch enabled
it while leaving default behavior unchanged. This implements shape-specialized **post-host-clamp** GDN settlement,
not device acceptance, mutable banks or a monolithic persistent arena.

Each turn owns up to eight graph IDs, one per retained width. Shape-aware MLX
compilation specializes verify widths and state geometry rather than baking
varying shape-derived integers into a shapeless graph. Every convolution state,
recurrent state and Q/K/V/g/beta/raw-QKV tape array is an explicit input. The
builder invokes the existing `GdnLayerTape::replay_into` arithmetic and returns
fresh convolution/recurrent handles for all GDN layers. The caller checks every
destination and full-attention frontier before invocation; unavailable bounded
plans leave caches unchanged and fall back to the established replay. Successful
construction rebinds state outputs and trims full-attention logical frontiers.
Draft append, host stop/cancellation decisions and evaluation boundaries retain
their existing behavior.

Plans are erased when their turn owner drops; lazy output roots retain their
inputs independently of that owner. This reduces repeated Rust graph building
to one compiled invocation after a plan has warmed. It does not remove replay
math/state writes or eliminate MLX's per-invocation output allocations. Per-turn
tracing cost is included in any end-to-end evaluation.

Added tests cover all 36 legal verify-width/keep combinations twice with fresh
input arrays, deferred evaluation after dropping the graph owner/input handles,
and non-mutating declines/errors. A stepper test compares exact logits, both
cache stacks, frontiers, history and cached continuation for keep 1 through 8,
including keep=1 followed by keep=8, draft-window wraps and subsequent AR
fallback. Run `cargo test -p mlx-core --lib compiled_commit -- --test-threads=1
--nocapture` on the coordinated Metal test worker. At this handoff, formatting
and whitespace checks have run; native build, GPU tests and matched performance
evaluation remain pending. The experiment stays opt-in until that evidence is
available.

Q retention remains unchanged: removing it safely also requires a state-only
sequential fallback or validated dummy-query fallback, crossing shared replay
APIs. Draft append remains unchanged because its dense chronological consumer
would retain or relocate tail copies unless attention/layout changes too; the
previous slower ring experiment is not repeated by this prototype.

The smallest measurement gate for mutable banks is a matched compiled-commit
off/on request with unchanged output/acceptance, plus a separate diagnostic run.
Existing `[mlx-compiled]` records whose function-ID high 16 bits equal `0xD51C`
identify these plans. Compare warm `hostSpanMs` separately from first-shape
traces, the complete `dspark_commit` host span (which also includes draft append),
and uninstrumented full-cycle time. A settlement-only follow-up can reuse frozen,
evaluated production-shape snapshot/tape inputs (48 layers, roughly 90 MiB of
logical state-plus-tape payload) and measure evaluate-to-completion while
counting allocator requests, cache hits, actual new-buffer misses and allocator
time. Bound unique live buffers, including roughly 75 MiB of fresh commit
outputs. Existing command `resourceBytes` is not an allocation counter.

Only material residual allocator cost after compiled replay supports a mutable
output-bank experiment. Predominantly cheap cache hits plus a small settlement
share of cycle latency would argue against it. A bank can avoid allocation
bookkeeping; the mandatory recurrence loads/stores remain.
