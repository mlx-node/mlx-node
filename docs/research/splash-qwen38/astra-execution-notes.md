# Execution architecture: bounded feasibility notes

Research-only inspection on September 22, 2026. The comparison target is the
mixed native GGUF `.cache/models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf`
and BF16 `.cache/models/qwen3.8-27b-dflash2` companion. Splash was verified clean
at `7e3c67e`. The current mlx-node worktree's existing changes were preserved.
No build, inference, or GPU benchmark was run for these notes.

Read `architecture-implementation.md` before this investigation. Segmented
attention is the current baseline. Eager state settlement, empty-finalization
skipping, and the other unproven experiments are not promoted here.

## What the two runtimes actually execute

**MLX already compiles the complete target verifier.** It is inaccurate to
describe its default path as rebuilding all verifier operations in Rust every
cycle. `crates/mlx-core/src/models/qwen3_5/model/forward.rs:392` routes the
speculative all-row forward through `forward_dflash2_compiled`; its contract at
line 496 includes embedding, every target layer, final normalization, and LM
head. The key includes model identity and verifier width; prefix length stays
shape-polymorphic. Live recurrent/conv state and attention K/V prefixes are
inputs, while taps, replay tapes, and new K/V are outputs. Cache publication
still occurs outside the compiled tape (`forward.rs:701-788`).

Compilation preserves a CPU graph, not encoded GPU commands:

- `crates/mlx-sys/mlx/mlx/compile.cpp:1026-1089` creates an unordered map of
  trace IDs to real arrays on every invocation, walks every tape entry, builds
  fresh input vectors/array descriptors, and calls `output_shapes` again for
  shapeless nodes. The numerical primitive objects are reused.
- `crates/mlx-sys/mlx/mlx/transforms.cpp:83-240` discovers the reachable graph
  and forms an evaluation tape. Its execution loop at lines 242-345 resolves
  cross-stream events/fences and calls each primitive's backend evaluation.
- `crates/mlx-sys/mlx/mlx/backend/metal/eval.cpp:51-104` calls `eval_gpu`, retains
  input/sibling storage, and may submit when command thresholds are reached.
- `crates/mlx-sys/mlx/mlx/backend/metal/device.cpp:394-517` binds resources,
  records hazards, inserts buffer barriers, and issues dispatches. Lines
  610-620 determine submission from dispatch count and element-based resource
  size units. Lines 746-765 create a concurrent compute encoder lazily.

**Splash's plans also are not pre-encoded Metal command buffers.**
`splash/runtime/ops/ExecutionPlans.hpp:89-140` holds immutable startup-selected
operator choices and workspace bounds, borrowed by models. Its
`splash/runtime/metal/CommandGraph.hpp:17-78` is an ordered CPU dispatch list
whose parameter bytes live in graph-owned storage. On every unconstrained
decode batch, `splash/runtime/model/Runtime.mm:2208-2266` builds a new list for
draft, verifier, policy, acceptance, recurrent commit, and draft-context commit.
`splash/runtime/metal/MetalBackend.mm:1310-1394` validates/prepares entries and
retains allocation owners; lines 1400-1439 create a new command buffer and one
ordinary compute encoder, bind every dispatch, and encode it again. Lines
1448-1500 register completion and submit after any sparse-map dependency.

The concrete difference is a deliberately bounded model-specific execution
schedule, stable arena storage, and device-readable acceptance/commit control.
It is not evidence that Splash skips all CPU encoding.

Three units must stay separate in reports:

1. A **command buffer** is a submitted completion/lifetime unit. MLX may submit
   several within one evaluation; Splash's ordinary unconstrained batch above
   submits one.
2. A **compute encoder** is an encoding scope inside a command buffer. MLX uses
   concurrent dispatch encoding plus explicit hazard barriers; Splash uses the
   ordinary encoder in the cited path. Encoder count is not dispatch count.
3. A **dispatch** launches one kernel grid. Both implementations issue many
   dispatches. Reducing command buffers alone neither fuses kernels nor removes
   their weight/KV traffic or recurrent arithmetic.

## The actual CPU/GPU boundary

For penalty-free greedy mlx-node decode,
`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:264-303` selects a
device-resident draft proposal. Its `verify_device` at lines 330-370 appends the
target verifier to that lazy graph without reading proposal IDs first.
`crates/mlx-core/src/engine/dspark_turn.rs:211-235` adds batched argmax, evaluates
it synchronously, then reads IDs and finds the accepted prefix on the host.
Thus the acceptance evaluation includes unfinished draft, verifier, argmax,
and any reachable previous-cycle state work. Its elapsed time is not the cost
of copying a handful of IDs or doing the host prefix comparison.

`crates/mlx-sys/mlx/mlx/transforms.cpp:287-311` may wait during evaluation when
active task/memory limits are reached. `eval()` also performs the final
completion wait at line 397, outside the current `eval_impl` trace span.
`device.cpp:733-743` similarly has a real `waitUntilCompleted` in explicit
stream synchronization. These waits are dependencies on outstanding GPU work;
they must not all be labeled removable CPU overhead.

After acceptance, `dspark_turn.rs:397-480` checks host cancellation, token
observers, EOS, and repetition in a simulated emission sequence **before**
computing the retained cache prefix. It excludes an accepted stop-causing token
from cache state, matching autoregressive continuation. Lines 1060-1095 then
commit with exact input provenance and optionally schedule the lazy commit.
`dflash2_decode.rs:422-437` explicitly replays recurrent state even on full
acceptance: verify carries FP32 state through the window, but accepted-state
replay must round through the model dtype after every token.

`splash/runtime/model/Runtime.mm:2244-2266` instead encodes acceptance and commit
after verifier policy in the same command. Its acceptance kernel reads proposed
and target IDs and produces retained count/next anchor
(`splash/runtime/metal/kernels/decode/sampling.metal:709-724,843-892`). The GDN
commit consumes `retained[batch]` on device
(`splash/runtime/metal/kernels/decode/gdn.metal:239-277`). Host finalization
validates results, swaps state parity, and advances logical lengths only after
completion (`Runtime.mm:1452-1525`).

This is not a drop-in numerical/state contract. Splash's cited GDN commit uses
FP32 recurrent-state pointers, while mlx-node's replay deliberately returns
model-dtype state with per-token rounding
(`crates/mlx-sys/src/mlx_gated_delta.cpp:472-575`). Importing Splash's recurrent
state policy would change the reference, even if aggregate throughput improved.

Cancellation contracts also differ. Splash records a pending failure for an
in-flight request (`splash/runtime/engine/Engine.cpp:108-122`) and drains its
command before suppressing output/cache publication (`Engine.cpp:1020-1038`).
mlx-node has per-token observer/cancel decisions before accepted-state commit.
An all-GPU commit followed by a host stop check would violate the current
mlx-node contract unless provisional state can be discarded/recomputed exactly.
Splash itself retains a host boundary for constrained output: its
`Runtime.mm:1538-1647` ticket submits draft, overlaps target verification with
grammar work, then submits a commit tail once host masks arrive.

## Feasibility of the larger designs

| Design                              | Concrete requirement                                                                                                                                       | What it can eliminate                                                                        | What remains                                                                                               |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Persistent verifier CPU plan        | Cache dependency indices and shape metadata rules, produce fresh MLX array identities                                                                      | Repeated dependency-ID hashing and temporary mapping setup                                   | Shape calculations, arrays, evaluator traversal, allocation, encoding, shader work                         |
| Native verifier dispatch plan       | Lower supported primitives into explicit dispatch descriptors and a lifetime-checked scratch layout                                                        | Generic graph construction/traversal, repeated operator routing, some allocation/bookkeeping | Fresh command submission, dynamic bindings/control, GPU arithmetic/traffic                                 |
| Pre-encoded dispatch sequence       | Supported indirect-command facilities or another documented reusable device command mechanism; stable resources, dynamic control buffers, explicit hazards | Some repeated host encoding                                                                  | Resource lifetime/visibility, submission/completion, shader work; capability fallback required             |
| GPU acceptance + provisional commit | Device keep count plus isolated next-state ownership and a host-authorized publication boundary                                                            | Host prefix comparison and potentially commit-build delay                                    | Host observers/cancel/stop semantics, commit correction on shorter prefix, final error/durability boundary |

A native plan cannot safely capture arbitrary `MTLBuffer*` values from one MLX
evaluation and replay them. MLX's temporary allocation, view offsets, donation,
and owner retention are evaluation-specific. The plan needs explicit slots for
weights, external inputs, transient scratch, and externally visible outputs.
Each in-flight cycle must own its binding table and scratch lease until GPU
completion and until downstream consumers release retained outputs. A cache
growth/reallocation must change the corresponding binding generation.

For this workload, fixed verifier width makes most projection geometry stable,
but prefix-length-dependent segmented attention still has real shape/stride,
split geometry, and workspace requirements. A plan must either rebind/reselect
these descriptors or key a compatible attention bucket. A single recorded
prefix-size launch is not a shapeless verifier. Mixed native GGUF formats must
retain their exact current reduction/rounding paths. Replacing them all with
Splash Q4 kernels is a format/numerical change, not execution-plan reuse.

Persistent storage needs a bounded arena budget based on actual runtime
dimensions, current allocator/device budgets, and pipeline limits. New launch
selection must use actual SIMD width, maximum threads, and available/static
threadgroup memory. Model-name or M5 routing is not an acceptable new key.
Fallback must remain valid on synthetic unsupported profiles.

For GPU commit, a safe first integration would keep the existing host clamp:
compute raw accepted count on the GPU, read the compact cycle result, run the
unchanged clamp, and pass the **authorized** keep count into a prepared commit
tail. That can reduce commit construction but does not remove the acceptance
wait. An optimistic tail before the host clamp would need separate provisional
state; on a shorter authorized prefix it must replay from the original snapshot,
not continue from the overcommitted state. Draft ring writes need equivalent
isolation, because modulo writes can overwrite old live rows. Do not publish
frontiers, parity, tokens, draft history, or session cache state until the
authorized state succeeds. Failed commands poison/discard the provisional
lease, never become a reusable session.

## Smallest proposed prototype: indexed compiled replay

This is a bounded experiment before implementing a new GPU scheduler. It is not
a prediction of a material speedup.

**Priority gate from the fresh candidate-9 diagnostic:** the parent investigation
reports 65 width-8 compiled invocations in
`.cache/benchmarks/splash-qwen38-phase5/current-command-trace*`: the first takes
13.188 ms; subsequent median is 0.8274 ms, maximum 8.689 ms; aggregate compiled
span is 81.4 ms. This wrapper span upper-bounds the narrower replay work. Even
removing it completely cannot reasonably explain an approximately twofold
cycle-time gap. The 336 evaluation spans total 14,451.94 ms, of which
13,240.94 ms is scheduler backpressure, with 8,449 task-wait sections and zero
memory-wait sections; these totals include load, warmup, and prefill and are not
decode-only attribution. They do not justify raising memory budgets.

Accordingly this prototype is a low-priority framework experiment, not the next
performance fix. Prioritize investigation of unnecessary GDN verifier output
work and production-shape SIMD decoding first, as identified by the parallel
state/kernel investigations. Retain execution plans as a possible way to apply
proven operator changes and remove measured exposed host work, after cost gates
are met. This update uses the parent investigator's diagnostic handoff; these
notes did not run another GPU job.

Extend `CompilerCache::CacheEntry` (`compile.cpp:303-314`) with a replay plan
built only after simplification/fusion succeeds (`compile.cpp:1137-1155`). Each
record stores the existing primitive reference, input slot indices, output slot
indices, sibling ordering, and static dtype/shape metadata. Store input,
constant/load, and final-output slots separately. The compiled tape owns the
same constants as today; cache erase/model teardown releases the plan with it
(`compile.cpp:370-375`; model/lifecycle.rs:891-906).

At invocation, allocate a dense per-call array-slot vector and execute records
in exactly the original `compile_replace` order. Retain fresh array descriptors
and the same dynamic `output_shapes` calls. Reuse the existing primitive
pointers; do not touch reductions, fusion policy, streams, GPU pipelines,
donation, scratch allocation, or host acceptance/stop behavior. Keep all slot
owners through replacement, matching the old map's lifetime. No shared mutable
per-cycle scratch belongs in the cached plan: nested invokes and separate
sessions must not overwrite each other's slots. Preserve constants that become
evaluated loads and duplicate/sibling outputs exactly. Retain an off switch.

This deliberately leaves evaluation traversal in place. A later direct
evaluation-tape path would have to splice in lazy external draft/cache
dependencies, preserve stream fences/events, and retain all outputs consumed
outside logits. Calling synchronous eval on inputs first to make that simpler
would add a boundary and invalidate the claimed comparison.

Acceptance checks for the prototype:

- Existing compile tests plus exact reference comparison for repeated values,
  shared/duplicate inputs, multi-output sibling primitives, captured weights,
  nested calls, load constants, and exceptions during first trace.
- Shapeless prefix changes across attention reduction boundaries; widths 1–8;
  different dtype/rank and stream cache keys; model destruction/reload with
  different weights; two sessions interleaved without stale arrays.
- Exact production output hash, accepted prefix per cycle, target attention
  frontier, recurrent/conv state, draft context/history, and cached continuation
  against the current baseline. Include each possible partial acceptance, full
  acceptance, EOS/observer/repetition/cancel inside the block, final-budget
  truncation, and continuation beyond the draft window.
- Ownership checks with async downstream consumers, dropped caller handles,
  allocator cache clearing, and forced short command limits. Peak residency
  must not become proportional to the number of past cycles.

Instrumentation and falsification:

1. Add CPU-only counters around warm-cache lookup, `compile_replace`/indexed
   replay, graph discovery, encoding excluding scheduler waits, scheduler
   backpressure, and final synchronous wait. Count nodes/edges, replay slots,
   allocations, dispatches, encoders, barriers, and submitted command buffers.
   The existing `mlx-compiled` span (`mlx_compiled_graph.cpp:169-179`) covers the
   wrapper plus compiled invocation, so it is an upper bound for this narrower
   replay optimization, not a measured replay-only cost.
2. Keep completion callbacks minimal and report CPU and GPU clocks separately.
   No per-dispatch forced waits. If the device cannot sample dispatch
   timestamps, report command/encoder-level envelopes and explicitly leave
   shader attribution unknown. Diagnostic throughput is not accepted speed.
3. Alternate clean processes for uninstrumented exact-workload timing after
   correctness. Match checkpoint/draft format, prompt/output, acceptance,
   resident cache state, token budget, and contention criteria.
4. Reject immediately on any changed output/state/lifetime failure. If warm
   replay CPU does not fall, reject the mechanism. If it falls but request/cycle
   wall time does not improve beyond paired variance, leave it disabled and
   classify the cost as hidden/too small. If the entire original compiled
   invocation span is much smaller than the measured per-cycle gap, this design
   cannot close that gap even under the impossible assumption that all of that
   span is removable and lies on the critical path.

The source provides no basis for promising that a persistent plan closes an
approximately twofold cycle-time gap. Plans do not reduce target projection
bytes, recurrent math, or attention bytes on their own. GPU acceptance moves a
small comparison and some scheduling decisions, while the existing readback
wait includes real deferred GPU work. The next measurement must bound exposed
CPU replay/encoding time before selecting a more invasive native plan.
