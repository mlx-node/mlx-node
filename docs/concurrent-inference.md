# Concurrent inference

How mlx-node behaves when several chat requests are in flight at once, why it
is not yet maximally efficient under that load, and what already exists in the
tree to change that. Findings are from an adversarially verified review of
`main` @ `2d1fe60e` (2026-08-08): 25 load-bearing claims, 19 confirmed, 6
corrected, 0 refuted. Stage 0's status below was updated after its robustness
work landed. The staged plans built on this doc:
`docs/superpowers/plans/2026-08-08-stage0-concurrency-robustness.md` (hazard
fixes) and `docs/superpowers/plans/2026-08-08-stage1-continuous-batching.md`
(batched decode → ragged step → hybrids, with the design provenance).

Two facts to carry into any batching work: only T=0 output is
schedule-invariant — T>0 sampling draws from the thread-local PRNG in row
order, so batch composition changes reproducibility (same as vLLM); and the
paged decode clear-cache interval is 1024 steps
(`PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT`, `array/memory.rs:97`), not the
64 some older notes claim.

## The short version

- Requests to the **same model** never overlap. They queue in a strict FIFO
  and each turn owns the model end to end — prefill, every decode step, and
  SSE delivery through terminal visibility or client disconnect.
- Requests to **different loaded models** genuinely run in parallel, one
  native thread per model, and MLX's thread-local design makes that safe.
- The **bottom of the paged stack — Metal kernels, FFI, block allocator — is
  already multi-sequence capable** (vLLM lineage). Every layer above it
  hard-codes batch = 1.

## The serialization ladder (same model)

```text
req A ─▶ withExclusive ─▶ model thread ─▶ whole turn: prefill → decode [1,1] × N → SSE (in lock)
              │
req B ─▶ ●────┘ parked until A's terminal SSE event (default cap: 16 waiters, then 429)
```

| Layer     | Mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Where                                                                                                                                         |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Server    | One rolling-promise FIFO per model object; `/v1/responses` and `/v1/messages` share it (aliases too). Held through terminal SSE visibility. When `res.write()` returns false, endpoint consumption parks until drain or transport close/error. A continuously backpressured live peer is torn down after `MLX_SSE_DRAIN_TIMEOUT_MS` (30s default), aborting native work and releasing the FIFO.                                                                                                                                                                                                                                                                                                                                                                                                                | `packages/server/src/session-registry.ts:1096-1137`, `streaming.ts`, `endpoints/responses.ts`, `endpoints/messages.ts` |
| Admission | Queue depth caps at 16 waiters by default (`DEFAULT_MAX_QUEUE_DEPTH_PER_MODEL`); over-cap requests get 429 + `Retry-After: 1`. Before a model is resident, `ModelWorkCoordinator.beginRequestLoadAdmission()` bounds cold-load requests with the same runner-plus-waiter capacity. Once a registry exists, two gates share one budget through ONE predicate (`assertAdmissionCapacity`): endpoint pre-dispatch permits cover host-mode writer waits and store-chain waits, and `withExclusive` checks every non-handed-off caller. A permit is retained through pre-lock awaits and consumed atomically at FIFO placement. Diagnostics report queued and pre-dispatch counts separately. Override via `maxQueueDepthPerModel` / `MLX_MAX_QUEUE_DEPTH_PER_MODEL`; config or host option `'unbounded'` opts out. | `model-work-coordinator.ts`, `session-registry.ts`, `server.ts`, both endpoint dispatch paths                          |
| Native    | One dedicated `"mlx-model"` OS thread per loaded model consumes one command — a whole turn — at a time. Chat NAPI fns only enqueue (streaming returns a handle immediately; non-streaming awaits a oneshot). Forwards never run on the tokio pool.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `crates/mlx-core/src/model_thread.rs:49-67`, `engine/cmd.rs:213-238`, `models/chat_napi.rs:85-215`                                            |
| Engine    | The decode loop is one token of one sequence: `y.item_at_int32(0)`, next forward `[1,1]`. The only batch>1 forwards are within a single request (prefill chunks, MTP/draft verification).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `engine/decode.rs:227-321`, `engine/mtp_turn.rs:367`                                                                                          |
| Host mode | Every request passes an exclusive writer bracket (`resolveModel`) before inference, even when the model is resident; single-resident by construction so it collapses into the same FIFO. Swaps drain all in-flight streams, then run a full serial unload/load; parked requests get 400 "binding changed"; there is no fast wrong-resident reject.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `endpoints/responses.ts:1781-1799`, `model-work-coordinator.ts:176-212`, `host/swap.ts`                                                       |

Correctness under interleaved sessions is protected (all-or-nothing prefix
verify, reset-and-reprefill on miss, single cache-owner bookkeeping):
alternating sessions is thrashy, never corrupt.

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

## Batch capability already in the tree

```text
HTTP mutex            ✗ serial
whole-turn engine     ✗ serial
decode loop [1,1]     ✗ serial
adapter (1 live req)  ✗ serial   ◀ the choke point
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
BlockAllocator        ✓ refcounts + prefix hash (vLLM-style pool)
FFI                   ✓ num_seqs = q.shape(0)
Metal kernels         ✓ block_tables [num_seqs, …], grid.y = sequence
```

- Kernels/FFI: `crates/mlx-paged-attn/metal/attention/paged_attention.metal:762-806`,
  `crates/mlx-sys/src/mlx_paged_ops.cpp:915`, launch grids in
  `crates/mlx-paged-attn/src/metal/paged_attention.rs:1112-1373`.
- Allocator: `crates/mlx-paged-attn/src/block_allocator.rs:67-129` (refcounted
  `PhysicalBlock`, `prefix_cache`, `find_longest_cache_hit`, `cache_full_blocks`);
  per-model-instance, not process-global.
- The choke: `PagedKVCacheAdapter` holds ONE live request; preparing a turn for
  a non-matching prompt releases the current one; decode metadata always emits
  `num_seqs = 1` (`transformer/paged_kv_cache_adapter.rs:1377-1390, 2203-2206, 3536`).

Latent scaffolding (designed for batching, never wired):

| Scaffolding                                                                    | State                                                                                                                                                                 |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PagedAttentionConfig.max_batch_size` ("for continuous batching", default 256) | dead — getter has zero callers (`config.rs:32-34,119-121`)                                                                                                            |
| `copy_blocks` copy-on-write kernel                                             | compiled into the metallib; no dispatcher, no callers (`metal/copy_blocks.rs`)                                                                                        |
| `SequenceBlockTable::replace_block` ("for copy-on-write")                      | test-only callers (`block_table.rs:63`)                                                                                                                               |
| Varlen kernel (`cu_seqlens_q`, N ragged sequences)                             | production always passes one sequence, `cu_seqlens_q=[0, q_len]` (`adapter:4062`)                                                                                     |
| Duplicated-row prefill layout (per-row heterogeneous `seq_lens`)               | env-gated escape hatches only: gemma4 `MLX_GEMMA4_PAGED_PREFILL_ROUTE=legacy`, lfm2 `MLX_LFM2_PAGED_PREFILL_PAGED_ATTENTION=1`; qwen3.5 always uses the varlen bridge |

Caveat for a future batched decode: the tuned grouped kernels (D256/D512
long-context routes) are gated `num_seqs == 1`
(`metal/paged_attention.rs:278,331`) and would need widening or fallback.

## Stage 0 concurrency hazards (verified status)

| #   | Hazard                                                         | Mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| H1  | Event-loop freeze after a streaming abort — **fixed**          | `resetCaches()` is async NAPI and waits on the model command channel without blocking Node. Streaming and synchronous chat turns install a turn cancel flag; flat/paged prefill, MTP/hidden-prefill helpers, and GDN materialized replay poll it at chunk boundaries and fail closed without publishing partial cache state. Residual: a single-shot prefill remains atomic, but a queued reset still parks only its promise, never the event loop. (`models/chat_napi.rs:77`, `engine/backend.rs:868`, family `turn_cancel` checkpoints.)                                                          |
| H2  | Non-streaming requests cannot be cancelled — **fixed**         | Public LM calls keep their ordinary names and accept the platform-native `AbortSignal`; `ChatSession` send paths pass `opts.signal`. Internally, the wrapper bridges to a two-phase native operation with one shared atomic flag. All supported session models, including Qianfan-OCR, poll it at safepoints. Both HTTP endpoints abort on disconnect and skip dispatch when the peer is already dead. Cancelled turns reject exactly with `"chat session cancelled"` and roll JS history back.                                                                                                                                                                                      |
| H3  | Unbounded queue by default — **fixed**                         | `createServer` defaults the per-model cap to 16 waiters; env/config/host options can override or explicitly select `'unbounded'`. The coordinator bounds arrivals during cold load before a `SessionRegistry` exists. After resolution, pre-dispatch permits and FIFO waiters share the same atomic budget. Over-cap requests get 429 + `Retry-After: 1` with separate queue/pre-dispatch diagnostics.                                                                                                                                                                                                                                                                               |
| H4  | SSE ignores backpressure — **fixed with bounded cancellation** | Endpoints stop pulling when `res.write()` returns false and wait close-safely for drain. A 30s default drain deadline converts a connected stalled peer into a sticky abort and destroys the transport, so the per-model FIFO cannot be held forever. Native TSFN delivery and the JS callback queue each have a 64-event ceiling; overflow cancels the turn. The model-thread `StreamTx` remains an unbounded implementation seam, but once the bounded bridge fills its receiver is dropped and the producer exits at a cooperative safepoint rather than growing for the lifetime of the connection. Stage 1 should replace this seam with a scheduler-owned bounded output ring. |

Cross-model residual coupling is perf-class, not correctness: the process-wide
Metal wired limit is set/restored per turn (`crates/mlx-core/src/stream.rs:142-262`),
and the flat decode path calls `clear_cache` every 256 steps, draining the
process-wide Metal free pool (`engine/backend.rs:146-149`, noted as
multi-model-hostile in `engine/cmd.rs:186-190`).

## Yardstick: vLLM v1

| Mechanism                                                                   | vLLM v1                                                                                   | mlx-node today                                                                                                                       |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Continuous batching (one shared forward per step; no prefill/decode phases) | core loop (`v1/core/sched/scheduler.py:440-451`)                                          | absent — whole-turn per request                                                                                                      |
| Per-step token budget + chunked prefill (TTFT fairness)                     | default on (2048 tokens/step budget)                                                      | absent — chunking exists per family but only bounds memory inside one turn                                                           |
| Live prefix sharing across running requests                                 | chained hashes + refcount + LRU, zero-copy                                                | half — same machinery exists but sharing is temporal (one live request); the SSD cold tier adds cross-process persistence vLLM lacks |
| Admission / preemption                                                      | `max_num_seqs`, watermark, preempt-and-recompute                                          | partial — per-model waiter cap, default 16 (429 + `Retry-After: 1`; pre-dispatch gate covers host-mode parking); no preemption       |
| Backend-agnostic scheduler                                                  | same scheduler on CPU (`platforms/cpu.py:164-166`); community vllm-metal runs it over MLX | n/a — evidence the design transfers                                                                                                  |

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
- **Stage 1 — phase-free continuous batching:** one scheduler token budget is
  allocated from each request's `num_computed_tokens → num_tokens` progress;
  uniform decode rows batch together while prefills consume bounded slices.
  Block-watermark admission defers work before allocation pressure. Start with
  dense qwen3 and keep hybrid stateful models in the exclusive lane.
- **Stage 2 — preemption and wider execution:** add recompute-first LIFO
  preemption with measured SSD escalation, then the ragged mixed-token step and
  per-request recurrent-state tables for hybrid families.
