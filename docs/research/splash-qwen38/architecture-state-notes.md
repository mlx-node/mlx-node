# DFlash2 cycle architecture and state settlement

This is a source audit, not a new benchmark. It compares the current mlx-node
DFlash2 cycle with Splash's runtime organization while keeping the model and
quantization differences explicit. No production code, build, or GPU/model run
was used for this note.

The main finding is that the current greedy path already has one proposal-plus-
verify materialization boundary. The CPU transfer at that boundary is tiny; the
boundary is expensive because it waits for the deferred GPU graph. The larger
remaining architecture opportunity is to schedule durable state settlement as
soon as the host stop clamp decides `keep`, so it can overlap CPU emission and
does not drift into the next cycle's measured wait. A Splash-style fully device-
resident acceptance and commit is possible only for a narrower request contract,
because mlx-node checks cancellation, an arbitrary token observer, extra stop
tokens, and repetition rules before it commits cache state.

## Actual mlx-node cycle and barriers

The engine's steady-state order is `propose -> verify -> accept -> stop-clamp ->
commit -> emit -> eval_boundary` (`crates/mlx-core/src/engine/dspark_turn.rs:823-1125`).
The stepper snapshots target caches and constructs the target forward/tapes in
`verify` or `verify_device`; neither method normally waits for the GPU
(`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:284-356`). The device
variant concatenates the anchor with the device-resident proposal, so there is
no proposal-boundary host read (`crates/mlx-core/src/engine/dspark_turn.rs:881-887`).

For greedy sampling without penalties, `logits.argmax` is constructed and
`argmax_arr.eval()` is the cycle's materialization point. It executes the lazy
proposal and verifier graph before proposal IDs and the target decisions are
read (`crates/mlx-core/src/engine/dspark_turn.rs:110-128`). `MxArray::eval`
ultimately calls `mlx_array_eval`; grouped synchronous evaluation calls
`mlx::core::eval` (`crates/mlx-core/src/array/data.rs:69-72,110-139`,
`crates/mlx-sys/src/mlx_nn_ops.cpp:85-96,131-166`). A scalar accessor also
calls `ensure_readable`, which evaluates an unavailable array before reading
its mapped data (`crates/mlx-sys/src/mlx_nn_ops.cpp:239-269,369-375`). The cost
seen at acceptance is therefore mainly the deferred GPU work and wait, not the
few bytes loaded by the CPU.

After that wait, the host stop clamp walks at most the accepted block. It checks
the output budget, then appends the simulated token, then checks cancellation,
the observer, EOS and extra EOS IDs, and repetition limits. It returns separate
`emit_count` and `keep` values so a tripping stop token may be emitted/history-
visible while its K/V slot is excluded (`crates/mlx-core/src/engine/dspark_turn.rs:289-380,932-963`).
This ordering is part of session correctness, not incidental control flow.

`commit` then restores the verifier snapshot to the accepted prefix and builds
recurrent-state replay plus draft-context append (`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:359-425`).
The normal replay path is already a fused recurrent kernel per GDN layer; its
fallback is a serial sequence of one-token steps
(`crates/mlx-core/src/models/qwen3_5/gated_delta.rs:256-297,333-378`). Replay
must also run on full acceptance: multi-row verification holds the recurrence
in FP32 across the window, while serial decoding stores/rounds state after each
token (`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:408-423`). Selecting
a verifier row or blindly retaining the final verifier state would change GGUF
decode numerics.

In steady state, `commit` mostly constructs lazy work. The engine performs CPU
history update, detokenization, and callbacks before `eval_boundary`, which
asynchronously schedules the target caches and boundary token
(`crates/mlx-core/src/engine/dspark_turn.rs:1058-1125`,
`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:433-436`,
`crates/mlx-core/src/models/forward.rs:181-190`). The next proposal consumes the
updated draft context, and the next verify consumes the updated target caches;
those data dependencies make the queued settlement execute before their
consumers. The next acceptance wait can consequently include prior-cycle
commit work. Calibration deliberately calls `materialize_adaptive_state()` to
keep that work out of the following probe, while the comments explicitly say
steady state remains asynchronous (`crates/mlx-core/src/engine/dspark_turn.rs:983-1014`,
`crates/mlx-core/src/engine/backend.rs:1903-1908`). Every 256 emitted tokens,
`synchronize_and_clear_cache` adds a full GPU synchronization and cache clear
(`crates/mlx-core/src/engine/dspark_turn.rs:1094-1098`,
`crates/mlx-core/src/array/memory.rs:52-68`).

Ending a cycle does not automatically make settlement synchronous. `finish`
moves the draft context back into model-owned state
(`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:427-430`); MLX arrays can
retain their deferred graph until a later continuation or persistence consumer
forces them. Any new eager scheduling design must retain those roots and surface
errors before the session is declared durable.

## CPU-visible data

For the measured greedy, no-penalty path at maximum draft length seven, the
cycle makes these token IDs CPU-visible after the single acceptance wait:

- target argmax decisions: eight `i32` values, 32 bytes
  (`crates/mlx-core/src/engine/dspark_turn.rs:115-128`);
- proposal IDs back-filled from the already evaluated device path: seven
  `i32` values, 28 bytes (`crates/mlx-core/src/engine/backend.rs:1812-1823`);
- verifier-token provenance copied again during commit: eight `i32` values,
  32 bytes (`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:386-400`).

That is at most 92 bytes of explicit ID data per cycle. The scalar reads access
mapped unified-memory storage; `to_int32` allocates a host vector and copies the
flattened array (`crates/mlx-core/src/array/data.rs:343-358`,
`crates/mlx-sys/src/mlx_common.h:325-349`). No verifier logits are copied to the
host. Eliminating the duplicate 32-byte provenance copy is reasonable cleanup,
but it does not remove the GPU evaluation boundary.

The sampled path has different traffic and synchronization. DFlash's non-greedy
selector evaluates and host-copies its top-16 candidate IDs and `L x 16 x 16`
score tensor before it walks the conditional path. At `L=7`, these contain 448
bytes of IDs and 7,168 bytes of FP32 scores
(`crates/mlx-core/src/models/qwen3_5/dflash2.rs:679-707`). Acceptance materializes
full-vocabulary FP32 target distributions on device, but the current source does
not copy those rows wholesale to the CPU. With DFlash's sparse proposal it reads
one target probability per attempted row, and rejection constructs and samples
the residual on device (`crates/mlx-core/src/sampling.rs:1003-1056,1224-1240`).
The generic dense-proposal path reads one target and one draft probability per
attempted row (`crates/mlx-core/src/sampling.rs:1178-1211`). RNG draws and
rejection decisions are sequential, so the greedy fused-decision design cannot
simply replace this path without reproducing its exact RNG consumption.

## What the phase timers mean

`DecodeProfiler` records wall time between `begin` and `end`; it does not assign
GPU kernels to phases (`crates/mlx-core/src/decode_profiler.rs:219-247`). In the
ordinary path, `dspark_verify` commonly measures lazy graph construction, while
`dspark_accept` contains `argmax_arr.eval()` and waits for proposal plus verify.
`dspark_commit` commonly measures construction and leaves execution for the
later async boundary or next dependent evaluation
(`crates/mlx-core/src/engine/dspark_turn.rs:869-978`). Optional instrumentation
inside `verify_device` can force logits there and therefore changes which phase
is charged (`crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs:350-355`).

Accordingly, the earlier informal 1-3% estimate is not a measured upper bound
on cycle-level synchronization or state-scheduling work. Host acceptance alone
is small, but its `eval` owns the wait for substantial deferred GPU work, and
commit work can be charged to the next cycle. No numeric gain should be assigned
to the architecture changes below without an end-to-end matched measurement.

## Splash's organization

Splash records draft, verify input, target forward/policy, acceptance, target GDN
commit, and draft-state commit into one `CommandGraph`, then submits it
asynchronously (`/Users/brooklyn/workspace/github/splash/runtime/model/Runtime.mm:2230-2261,2292-2299`).
The completion ticket waits once, after which CPU finalization reads generation,
retained and accepted counts, the next anchor, and only the retained output
tokens (`Runtime.mm:49-83,1459-1523`). With one lane and at most seven proposal
tokens, that is four 32-bit scalars plus at most seven 32-bit output tokens, at
most 44 bytes read by the CPU.

The retained count remains device-resident while Splash's GDN and draft-state
commit kernels run (`Runtime.mm:1332-1435`). Target recurrent state uses explicit
current/next buffers and CPU finalization swaps the parity only after completion
(`Runtime.mm:1272-1279,1500-1523`). Its acceptance kernel handles the generation
budget and two stop tokens on-device
(`/Users/brooklyn/workspace/github/splash/runtime/ops/Sampling.cpp:209-235`,
`/Users/brooklyn/workspace/github/splash/runtime/metal/kernels/decode/sampling.metal:709-760,843-893`).

This is a stronger contract than moving argmax to a tiny kernel: the commit
consumes a device count and is already encoded before the CPU sees results.
It also has narrower stop semantics than mlx-node. mlx-node's observer can stop
on arbitrary token-dependent application state; cancellation can arrive during
the cycle; extra EOS and repetition checks depend on full host history. Committing
before those checks would retain cache slots that current code deliberately
excludes. Delaying cancellation to the next cycle would be a user-visible
semantic change, not a scheduling implementation detail.

Splash's backend normally emits the graph through one compute encoder and uses
an asynchronous command-buffer completion handler
(`/Users/brooklyn/workspace/github/splash/runtime/metal/MetalBackend.mm:1414-1497`).
Its development per-dispatch profiler instead serializes dispatches and waits,
so those timings perturb normal scheduling (`MetalBackend.hpp:336-353`,
`MetalBackend.mm:1291-1309`). A local M5 Max capability probe found timestamp
counter sampling supported only at stage boundaries, not dispatch boundaries:
`.cache/benchmarks/splash-qwen38-phase3/counter-capabilities.log` records
`dispatch=0`, `stage=1`, `set=timestamp`, and successful timestamp-buffer
creation. Per-dispatch `MTLCounterSampleBuffer` insertion is therefore not a
viable low-perturbation diagnostic on this machine. Splitting encoders or command
buffers would itself change dispatch/fence behavior.

## Bounded implementation routes

### 1. Schedule settlement before CPU emission

After `clamp_dspark_cycle` computes `keep`, have `commit` return or expose the
target-cache and draft-context roots, and call `MxArray::async_eval_arrays` on
them immediately. Then perform history updates, detokenization, and streaming
callbacks while the GPU settles state. Today that work is not explicitly queued
until after emission at `eval_boundary`.

This preserves the existing acceptance barrier, host stop semantics, recurrence
arithmetic, and exact retained prefix. It changes scheduling rather than math.
The implementation needs an owned pending-settlement object so arrays and
snapshots remain alive, and a synchronous error checkpoint before final session
publication. The next cycle may consume the pending roots directly; stop/error
exit must also force or cancel publication consistently. Test CPU callbacks that
block, callbacks that request stop, cancellation at each accepted position,
full and partial acceptance, and a cached second turn.

### 2. Remove redundant provenance materialization

The engine already owns `[anchor, draft_ids...]` after acceptance. Pass those
verified IDs into `commit`, or retain an equivalent host vector in the stepper,
instead of converting `verified_ids_device` again. This removes the final
32-byte copy and simplifies lifetime/error handling. It is a bounded cleanup,
not the main cycle optimization. Tests must cover device and host proposals,
proposal truncation, zero-draft AR-through-verify, and mismatch rejection.

### 3. Prebuild the eight commit shapes

`keep` is bounded to one through eight. Compile or cache a settlement graph for
each retained width, then bind the current snapshot/tape/context roots after the
host clamp. This can move MLX graph construction out of the critical post-wait
section while retaining the current CPU decision. The difficult part is dynamic
cache ownership: compiled graphs must not capture stale arrays, alias donated
state, or advance logical length before successful completion. Compare every
cache array, provenance history, and the next proposal with serial decode across
all keep widths.

### 4. Device-count commit for a restricted fast lane

For requests with greedy sampling, no penalties, only the supported EOS set, no
token observer, no repetition cutoff, and an explicitly cycle-boundary cancel
contract, a custom acceptance kernel could produce output IDs and retained count
for GDN and draft-context commit in the same device graph. The CPU would wait
once after settlement, like Splash. A general implementation still needs the
host clamp; it could write the clamped count back to a tiny shared buffer and
submit a second settlement command, which removes graph specialization but not
the acceptance wait.

The fast lane must be selected fail-closed. It must reproduce greedy tie/NaN/
signed-zero behavior, force-think-end, budget handling, terminal-anchor rules,
and cancellation semantics. If cancellation remains observable inside a cycle,
the fast lane needs rollback/healing before it can publish state.

### 5. Explicit ping-pong state owner

A larger rewrite could give DFlash2 imperative current/next GDN and draft-context
buffers. Verification and settlement write the inactive side; only successful
completion swaps parity and advances logical length/history. This matches
Splash's atomic publication model and is the foundation for a reusable command
graph, but it crosses MLX cache ownership, persistence, paged-session provenance,
OOM/error rollback, and continuation boundaries. It should follow the earlier
routes only if profiles show state scheduling or graph construction material.

## Required proof before retaining a change

Numerical tests must compare every target cache array, every draft-context cache
array, recurrent state, convolution state, logical frontiers, provenance token
history, emitted IDs, and the next ordinary proposal. Cover keep widths 1-8,
full acceptance, first-row rejection, stop on an accepted draft, stop on the
boundary, extra EOS, budget exhaustion, repetition cutoffs, observer stop,
cancellation at each clamp position, force-think-end, AR fallback, and both
greedy and sampled paths. Re-run a cached continuation and a cold replay of the
same conversation. GDN comparisons must be exact for the retained GGUF path;
tests that only compare final logits can miss state drift that appears next turn.

Performance validation must use an uninstrumented matched end-to-end cohort and
retain prompt IDs, output hashes, acceptance counts, cache state, and full cycle
wall time. Stage-boundary command-buffer GPU timestamps may help separate broad
submissions, but per-dispatch attribution on this M5 requires a perturbing
profiling path and must be labeled accordingly.
