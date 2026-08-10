# Concurrent inference

How mlx-node behaves when several chat requests are in flight at once and how
the dense-Qwen3 continuous-batching lane is built. The original findings came
from an adversarially verified review of `main` @ `2d1fe60e` (2026-08-08): 25
load-bearing claims, 19 confirmed, 6 corrected, 0 refuted. This document now
describes the Stage 0 robustness work and the landed Stage 1 scheduler. The
staged plans built on the original review:
`docs/superpowers/plans/2026-08-08-stage0-concurrency-robustness.md` (hazard
fixes) and `docs/superpowers/plans/2026-08-08-stage1-continuous-batching.md`
(batched decode → ragged step → hybrids, with the design provenance).

Two facts to carry into any batching work: only T=0 output is
schedule-invariant — T>0 sampling draws from the thread-local PRNG in row
order, so batch composition changes reproducibility (same as vLLM); and the
paged decode clear-cache interval is 1024 steps
(`PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT`, `array/memory.rs:97`), not the
64 some older notes claim.

## Current status

- Different sessions on one eligible dense Qwen3 paged model may overlap. The
  server admits up to the native scheduler's sequence capacity (8 by default),
  and the model thread advances them together. A single `ChatSession` still
  allows only one turn in flight.
- Flat-cache, multimodal, speculative, hybrid-family, training, save, and reset
  commands stay in the byte-compatible exclusive/barrier lanes. A forced-serial
  A/B switch keeps the old whole-turn path available.
- Different loaded models continue to run in parallel, one native model thread
  per model.

```text
request A ─┐
request B ─┼─▶ SessionRegistry.withAdmission(max_num_seqs)
request C ─┘           │
                       ▼
              one "mlx-model" OS thread
                       │
                       ▼
          phase-free, decode-first StepPlan
          ├─ uniform decode rows ─▶ one [N,1] forward
          └─ remaining budget    ─▶ pinned prefill slice
```

There are no persistent prefill and decode queues. Each row only records
`num_computed_tokens` and `num_tokens`; its work kind is derived while building
the next plan. Decode-first is a priority inside that one plan, keeping
interactive rows moving before a long newly admitted prefill consumes the
step's remaining token budget.

| Layer         | Current mechanism                                                                                                                                                                                                                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Server        | `/v1/responses` and `/v1/messages` select `withAdmission` only when a paged model explicitly reports a capacity greater than one. Every other model uses the existing `withExclusive` lane. Both routes retain their admission slot through terminal visibility or disconnect.                                              |
| Admission     | `SessionRegistry` is a FIFO counting semaphore for the batched lane. Active, queued, and pre-dispatch requests share the existing bounded budget; overflow still returns 429 plus `Retry-After: 1`. The default is 16 queued requests unless config or `MLX_MAX_QUEUE_DEPTH_PER_MODEL` overrides it or selects `unbounded`. |
| Native thread | One scheduler and executor live beside the model on its dedicated OS thread; `MxArray` never crosses threads. Idle waits block, while busy periods poll commands only between steps.                                                                                                                                        |
| Scheduler     | One global token ceiling serves decode rows first, then pinned prefill slices. Exclusive commands run only with an empty running set; reset/generate/save/train are barriers. Block watermark and already-promised growth can defer admission before allocation pressure.                                                   |
| Cache         | `PagedKVCacheAdapter` owns a request table keyed by sequence id over one refcounted block pool. Live requests can share verified prefix blocks. SSD restore is asynchronous: `WaitingForSsd` rows park while runnable peers continue.                                                                                       |
| Executor      | Dense Qwen3 performs one batched `[N,1]` decode forward for eligible rows and a bounded prefill dispatch when budget remains. Each row keeps its own history, penalties, sampling config, stop state, stream sink, and cancellation snapshot.                                                                               |
| Transport     | SSE writes honor Node backpressure and stop native pulls until drain or close. The 30-second default drain deadline aborts a connected stalled peer and releases admission.                                                                                                                                                 |

Per-request prefix verification remains all-or-nothing. A miss releases and
rebuilds only that sequence's slot; it never resets a peer. Batched admission
uses fresh JS sessions while retaining the native refcounted prefix cache, so
correctness is independent of the warm single-slot registry.

## Scheduler knobs

The native knobs are read once per process:

| Knob                            | Default | Meaning                                                                             |
| ------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `MLX_SCHED_MAX_NUM_SEQS`        | `8`     | Native running-set cap and server admission capacity (hard-clamped to 32).          |
| `MLX_SCHED_MAX_BATCHED_TOKENS`  | `2048`  | Total tokens planned in one scheduler step.                                         |
| `MLX_SCHED_LONG_PREFILL_TOKENS` | `2048`  | Maximum prefill progress for one request in one step.                               |
| `MLX_SCHED_WATERMARK_FRACTION`  | `0.05`  | Free-block headroom retained while work is already live.                            |
| `MLX_SCHED_RESERVE_FULL_ISL`    | `1`     | Reserve each admitted request's remaining prompt growth in the must-fit test.       |
| `MLX_PAGED_PER_SEQ_CTX`         | `32768` | Per-sequence context used by the pool budget formula.                               |
| `MLX_SERVE_FORCE_SERIAL`        | `0`     | Route eligible Qwen3 turns through the legacy whole-turn path for A/B and rollback. |

Two reproducibility rules are deliberate:

1. Only greedy `temperature = 0` output is schedule-invariant. With sampling,
   each row draws from the model thread's PRNG in row order, so changing batch
   composition can change output, as it does in vLLM.
2. Each request's legal prefill break-set is pinned at admission. The shared
   budget decides when a pinned slice runs, never where it is split; this
   preserves family-specific chunk-boundary invariants.

Outside the chat engine two families break the threading pattern: Harrier
embeddings run forwards on tokio's blocking pool (`models/harrier/model.rs:104-177`),
and `pp_*` document models run forwards inside synchronous NAPI calls — on the
Node event loop itself (`models/pp_text_det/model.rs:50`).

## Why serial turns are the wrong shape for concurrent load

Decode on Apple Silicon is weight-bandwidth-bound: every step streams the full
weights from unified memory. Serving N users serially spends N full weight
streams to advance each user by one token — 4 users see ~¼ speed each plus
queue-length TTFT (a 32k prefill blocks every other session for its full
duration). One forward per step over N sequences reads the weights once; the
win is near-linear until KV reads/compute saturate, and is largest at exactly
the realistic 2–8 concurrent range.

## Batch-capability boundary

```text
server admission       ✓ counting semaphore for eligible paged models
phase-free scheduler   ✓ token budget, watermark, occupancy histogram
Qwen3 adapter          ✓ per-sequence request table + shared prefix blocks
Qwen3 executor         ✓ one uniform [N,1] decode forward
BlockAllocator         ✓ refcounts + prefix hash (vLLM-style pool)
FFI / Metal kernels    ✓ num_seqs = q.shape(0), grid.y = sequence
ragged mixed step      △ Stage 2, behind StepExecutor seam
hybrid recurrent state △ Stage 1.5/2, currently exclusive
```

- Kernels/FFI: `crates/mlx-paged-attn/metal/attention/paged_attention.metal:762-806`,
  `crates/mlx-sys/src/mlx_paged_ops.cpp:915`, launch grids in
  `crates/mlx-paged-attn/src/metal/paged_attention.rs:1112-1373`.
- Allocator: `crates/mlx-paged-attn/src/block_allocator.rs:67-129` (refcounted
  `PhysicalBlock`, `prefix_cache`, `find_longest_cache_hit`, `cache_full_blocks`);
  per-model-instance, not process-global.
- The adapter request table and batched metadata are in
  `crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs`; the dense Qwen3
  executor is in `models/qwen3/model.rs`.

Remaining scaffolding:

| Scaffolding                                                      | State                                                                                                                                                                 |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `copy_blocks` copy-on-write kernel                               | compiled into the metallib; no dispatcher, no callers (`metal/copy_blocks.rs`)                                                                                        |
| `SequenceBlockTable::replace_block` ("for copy-on-write")        | test-only callers (`block_table.rs:63`)                                                                                                                               |
| Varlen kernel (`cu_seqlens_q`, N ragged sequences)               | used by family-specific single-request paths; not yet the Qwen3 mixed prefill+decode scheduler step                                                                   |
| Duplicated-row prefill layout (per-row heterogeneous `seq_lens`) | env-gated escape hatches only: gemma4 `MLX_GEMMA4_PAGED_PREFILL_ROUTE=legacy`, lfm2 `MLX_LFM2_PAGED_PREFILL_PAGED_ATTENTION=1`; qwen3.5 always uses the varlen bridge |

The tuned grouped D256/D512 long-context kernels remain gated to
`num_seqs == 1`; dense Qwen3 head dimension 128 uses the generic batched route.

## Stage 0 concurrency hazards (verified status)

| #   | Hazard                                                         | Mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1  | Event-loop freeze after a streaming abort — **fixed**          | `resetCaches()` is async NAPI and waits on the model command channel without blocking Node. Streaming and synchronous chat turns install a turn cancel flag; flat/paged prefill, MTP/hidden-prefill helpers, and GDN materialized replay poll it at chunk boundaries and fail closed without publishing partial cache state. Residual: a single-shot prefill remains atomic, but a queued reset still parks only its promise, never the event loop. (`models/chat_napi.rs:77`, `engine/backend.rs:868`, family `turn_cancel` checkpoints.)                                                                                                                        |
| H2  | Non-streaming requests cannot be cancelled — **fixed**         | Public LM calls keep their ordinary names and accept the platform-native `AbortSignal`; `ChatSession` send paths pass `opts.signal`. Internally, the wrapper bridges to a two-phase native operation with one shared atomic flag. All supported session models, including Qianfan-OCR, poll it at safepoints. Both HTTP endpoints abort on disconnect and skip dispatch when the peer is already dead. Cancelled turns reject exactly with `"chat session cancelled"` and roll JS history back.                                                                                                                                                                   |
| H3  | Unbounded queue by default — **fixed**                         | `createServer` defaults the per-model cap to 16 waiters; env/config/host options can override or explicitly select `'unbounded'`. The coordinator bounds arrivals during cold load before a `SessionRegistry` exists. After resolution, pre-dispatch permits and FIFO waiters share the same atomic budget. Over-cap requests get 429 + `Retry-After: 1` with separate queue/pre-dispatch diagnostics.                                                                                                                                                                                                                                                            |
| H4  | SSE ignores backpressure — **fixed with bounded cancellation** | Endpoints stop pulling when `res.write()` returns false and wait close-safely for drain. A 30s default drain deadline converts a connected stalled peer into a sticky abort and destroys the transport, so an admission slot cannot be held forever. Native TSFN delivery and the JS callback queue each have a 64-event ceiling; overflow cancels the turn. The model-thread `StreamTx` remains an unbounded implementation seam, but once the bounded bridge fills its receiver is dropped and the producer exits at a cooperative safepoint rather than growing for the lifetime of the connection. A scheduler-owned bounded output ring remains future work. |

Cross-model residual coupling is perf-class, not correctness: the process-wide
Metal wired limit is set/restored per turn (`crates/mlx-core/src/stream.rs:142-262`),
and the flat decode path calls `clear_cache` every 256 steps, draining the
process-wide Metal free pool (`engine/backend.rs:146-149`, noted as
multi-model-hostile in `engine/cmd.rs:186-190`).

## Yardstick: vLLM v1

| Mechanism                                                                   | vLLM v1                                          | mlx-node today                                                                                             |
| --------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Continuous batching (one shared forward per step; no prefill/decode phases) | Core scheduler loop                              | Dense Qwen3 uniform decode; other families remain exclusive                                                |
| Per-step token budget + chunked prefill                                     | Enabled by default                               | One 2048-token ceiling with decode-first planning and pinned per-request breaks                            |
| Live prefix sharing                                                         | Chained hashes, refcounts, and LRU               | Refcounted block pool across live Qwen3 sequences, plus an asynchronous SSD cold tier                      |
| Admission / preemption                                                      | `max_num_seqs`, watermark, preempt-and-recompute | Sequence cap and reserve-aware watermark admission; defers instead of preempting today                     |
| Backend seam                                                                | Scheduler drives platform-specific workers       | `StepExecutor` separates scheduler policy from dense-Qwen3 execution; ragged and hybrid executors are next |

What does not transfer: CUDA-graph padding (MLX lazy graphs play that role),
the multi-process ZMQ split (a scheduler thread suffices at this scale),
async-scheduling depth > 1 (one Metal command queue + lazy eval → one-step
overlap at best), and "use all GPU memory for KV" pool sizing (on unified
memory the pool competes with the weights — see
`docs/architecture.md` "Unified memory decides the cache hierarchy").

## Direction

- **Stage 0 — robustness (landed):** H1–H4 are regression-locked: ordinary
  non-streaming APIs with `AbortSignal`, pre-dispatch disconnect checks,
  fail-closed chunk-boundary prefill cancellation, async reset, cold-load and
  resident admission caps, and deadline-bounded SSE backpressure.
- **Stage 1 — dense Qwen3 continuous batching (landed):** the phase-free
  scheduler, uniform batched decode, per-row epilogue, live prefix sharing,
  block-watermark admission, asynchronous SSD restore, server semaphore, and
  same-binary forced-serial rollback path are implemented.
- **Stage 1.5 — hybrid entry:** give LFM2 a per-request convolution-state table
  and admit it to the batched lane without changing the scheduler policy.
- **Stage 2 — preemption and wider execution:** add recompute-first LIFO
  preemption with measured SSD escalation, swap in a ragged mixed-token
  executor behind `StepExecutor`, then debit per-request recurrent-state tables
  for GDN and Gemma4 against the same admission budget.
