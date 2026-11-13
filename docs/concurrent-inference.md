# Concurrent inference

How mlx-node behaves when several chat requests are in flight at once and how
the dense and hybrid continuous-batching lanes are built. The original findings came
from an adversarially verified review of `main` @ `2d1fe60e` (2026-08-08): 25
load-bearing claims, 19 confirmed, 6 corrected, 0 refuted. This document now
describes the Stage 0 robustness work and the landed Stage 1 scheduler. The
staged plans built on the original review:
`docs/superpowers/plans/2026-08-08-stage0-concurrency-robustness.md` (hazard
fixes) and `docs/superpowers/plans/2026-08-08-stage1-continuous-batching.md`
(batched decode → ragged step → hybrids, with the design provenance).

Request isolation does not imply bit-identical output across batch shapes:
GEMV/GEMM reduction changes can flip greedy decisions. Scheduled speculative
requests own their RNG streams; ordinary AR sampling still draws from the
model thread's PRNG in row order. The paged decode clear-cache interval is
1024 steps (`PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT` in `array/memory.rs`).

## Current status

- Different sessions on one eligible Qwen3, LFM2, Qwen3.5 dense/MoE, Gemma4, Muse-Glimmer,
  or NemotronH paged model may overlap. The server admits up to the native scheduler's physical
  sequence capacity, and the model thread advances them together. A single
  `ChatSession` still allows only one turn in flight.
- Flat-cache, training, save, and reset commands stay in exclusive/barrier
  lanes. Gemma4 ordinary text rows use grouped full/sliding paged KV and fused
  decode. Fixed-depth text DSpark also shares target verification waves, with
  request-owned draft contexts and RNG. Explicit greedy adaptive DSpark also
  shares verification waves when no fixed confidence threshold is requested.
  Media, assistant drafts and sampled adaptive speculation retain ordered lanes;
  reset and stats commands are barriers. Qwen3.5 dense/MoE text turns are
  scheduled by default on up to 4 lanes, computed from the machine's memory
  at load; up to 8 on machines with 256 GB or more (see "Qwen3.5 decode
  lanes" below); fixed-depth native MTP uses
  per-owner GDN replay and
  draft KV history there, while adaptive MTP, DFlash2 and media turns stay
  exclusive.
  Muse DFlash packing is experimental (`MLX_SCHEDULED_DFLASH=1`); its default
  remains the existing flat speculative lane. Loading a draft leaves ordinary
  batched owners available on the resident target.
- Different loaded models continue to run in parallel, one native model thread
  per model.

### Qwen3.5 decode lanes

A Qwen3.5 dense or MoE load runs up to 4 lanes, computed from the machine's
memory at load; up to 8 on machines with 256 GB or more. The ceiling
(`qwen35_lane_ceiling`) reads physical memory (`hw.memsize`), not the Metal
working set: 8 (`HYBRID_LIVE_STATE_UNITS`) from 256,000,000,000 bytes, which a
256 GB Mac (256 GiB) passes and a 192 GB Mac (192 GiB) does not; 4 below it
or when the memory is unreadable. `MLX_SCHED_MAX_NUM_SEQS` only lowers it.
The memory plan (`evaluate_qwen35_plan`) first plans one lane and its window.
It then grants the most lanes, up to that ceiling, whose extra state (each
lane's GDN state and, on an MTP load, its draft state at the full window) and
joint decode step (one AR row, or 8 MTP verify rows, per lane; on a MoE load
each row's top-k expert work besides) keep that window. Lanes give way first: the window is
never made smaller for a lane, and a load the one-lane plan admits always
loads. A load whose paged cache persists to the cold tier also prices, for
every lane, the first included, the 2 GDN states of checkpoint rungs a
sequence may keep past a refused claim until its turn-end sidecar capture.
A scheduled turn reserves those 2 states after its GDN cell only when the
memory governor has the room at once; a turn without that reservation keeps
no refused rung, and the reservation never makes a turn wait or fail.
`MLX_SERVE_FORCE_SERIAL=1` plans one lane. The granted count is the
only source of the recurrent-state table width, the admission check, the
scheduler's running-set capacity and `maxConcurrentSequences()`; the memory
plan's log names it (`N decode lanes`).

Unit-test table (`lanes_follow_the_device`; physical memory = the column,
working set = 3/4 of it; one lane's window in tokens):

| Model (weights)              | MTP | 16 GB      | 24 GB       | 36 GB       | 64 GB   | 128 GB | 192 GB | 256 GB | 512 GB |
| ---------------------------- | --- | ---------- | ----------- | ----------- | ------- | ------ | ------ | ------ | ------ |
| 0.8B bf16 (1,666 MiB)        | off | 4          | 4           | 4           | 4       | 4      | 4      | 8      | 8      |
| 0.8B bf16 (1,666 MiB)        | on  | 4          | 4           | 4           | 4       | 4      | 4      | 8      | 8      |
| 9B 8-bit (9,500 MiB)         | off | 2 (28,665) | 4           | 4           | 4       | 4      | 4      | 8      | 8      |
| 9B 8-bit (9,500 MiB)         | on  | 1 (4,089)  | 1 (244,389) | 3           | 4       | 4      | 4      | 8      | 8      |
| 27B (16,000 MiB)             | off | no load    | no load     | 1           | 4       | 4      | 4      | 8      | 8      |
| 27B (16,000 MiB)             | on  | no load    | no load     | 1 (188,409) | 4       | 4      | 4      | 8      | 8      |
| 35B-A3B 4-bit (18,780 MiB)   | off | no load    | no load     | 4           | 4       | 4      | 4      | 8      | 8      |
| 35B-A3B 4-bit (18,780 MiB)   | on  | no load    | no load     | 1           | 4       | 4      | 4      | 8      | 8      |
| 122B-A10B 4-bit (65,440 MiB) | off | no load    | no load     | no load     | no load | 4      | 4      | 8      | 8      |
| 122B-A10B 4-bit (65,440 MiB) | on  | no load    | no load     | no load     | no load | 4      | 4      | 8      | 8      |

Cells are granted lanes; the window is 262,144 unless given in brackets. With
the cold tier on, five cells change: 9B MTP off at 16 GB is 1 (24,569), 9B
MTP on is 1 (1,760) at 16 GB and 1 (241,173) at 24 GB, and 27B at 36 GB is
1 (258,041) with MTP off and 1 (180,217) with MTP on. A
window below 262,144 means the one-lane plan already spent the device's
memory on context, so further lanes are granted only when their cost fits
without lowering it.

```text
request A ─┐
request B ─┼─▶ SessionRegistry.withAdmission(max_num_seqs)
request C ─┘           │
                       ▼
              one "mlx-model" OS thread
                       │
                       ▼
          phase-free, decode-first StepPlan
          ├─ default ─▶ uniform decode + bounded prefill
          └─ ragged  ─▶ one packed varlen forward
```

There are no persistent prefill and decode queues. Each row only records
`num_computed_tokens` and `num_tokens`; its work kind is derived while building
the next plan. Decode-first is a priority inside that one plan, keeping
interactive rows moving before a long newly admitted prefill consumes the
step's remaining token budget.

| Layer         | Current mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Server        | `/v1/responses` and `/v1/messages` select `withAdmission` only when a paged model explicitly reports a capacity greater than one. Every other model uses the existing `withExclusive` lane. Both routes retain their admission slot through terminal visibility or disconnect.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Admission     | `SessionRegistry` is a FIFO counting semaphore for the batched lane. Active, queued, and pre-dispatch requests share the existing bounded budget; overflow still returns 429 plus `Retry-After: 1`. The default is 16 queued requests unless config or `MLX_MAX_QUEUE_DEPTH_PER_MODEL` overrides it or selects `unbounded`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Native thread | One scheduler and executor live beside the model on its dedicated OS thread; `MxArray` never crosses threads. Idle waits block, while busy periods poll commands only between steps. Qwen3, LFM2/2.5, Qwen3.5 Dense/MoE, Gemma4, Muse-Glimmer, and NemotronH instantiate the same engine-owned `HybridSchedulerState<B>`; their backends provide cache/recurrent/prefix/decode hooks rather than duplicating request lifecycle code.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Scheduler     | One global token ceiling serves decode rows first, then pinned prefill slices. Exclusive commands run only with an empty running set; reset/generate/save/train are barriers. Block growth and hybrid recurrent-state bytes share one unified-memory admission watermark; an actual lazy-allocation squeeze preempts exactly one newest running victim.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Cache         | `PagedKVCacheAdapter` owns a request table keyed by sequence id over a refcounted block pool. Hybrid families add one manager per compatible KV group. Gemma4 owns distinct full/sliding pools; expired sliding blocks become null sentinels so logical positions remain absolute. A preempted victim releases its live tables and replays safely. Sidecar-free SSD reads can park in `WaitingForSsd`; hybrid exact-boundary restores currently reconcile their K/V and companion state synchronously.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Executor      | Dense Qwen3 defaults to uniform `[N,1]` decode plus bounded prefill; `MLX_SCHED_RAGGED_STEP=1` instead packs every planned slice. LFM2 defaults to a row-exact decode wave — one N=1 forward per row inside the shared step, bit-identical to serial — because the fused `[N,1,H]` path's M=N GEMM tiles round differently than M=1 and the carried ShortConv state amplifies a few bf16 ULP into occasional greedy near-tie flips; `MLX_LFM2_FUSED_BATCH_DECODE=1` opts back into one fused forward (stacking/scattering private ShortConv rows) for benchmarking. Text-only Qwen3.5 dense and MoE stack/scatter private GDN rows; MoE AR decode routes experts over the same `[N,1,H]` tensor rather than falling back to one forward per row. A MoE MTP verify step runs every projection, GDN layer, expert block and the LM head on each owner's own `[1,W]` rows (one paged attention kernel still serves all rows): a packed `[N*W]` matmul picks another kernel and rounding by row count, and a one-ulp change in a router input moves a row to other experts. Its tokens equal each lane run alone, but each verify step streams the non-expert weights once per owner, N weight streams (Qwen3.6-35B-A3B: about 1.94 B parameters a pass). NemotronH has a fused lane — stacking/scattering per-request Mamba-2 rows (conv `[3,6144]` + SSM `[64,64,128]` f32 per SSM layer) around each SSM layer, routing its pure MoE-FFN layers over the shared `[N,1,H]` tensor, and driving its six GQA layers through the batched paged kernels — but that lane is reachable only on a **dense/bf16** checkpoint. `load_inner` (`models/nemotron_h/persistence.rs`) sets `row_exact_decode_projections` for **any** quantized checkpoint (a top-level quant mode or any per-layer override), which the only published NemotronH checkpoint (NVFP4 30B-A3B) always is, so the decode wave instead dispatches to `run_row_exact_decode_wave` (`models/nemotron_h/model.rs`), one single-row decode per row with the logits concatenated — bit-identical to N serial decodes, but N full weight streams. On that checkpoint the batching win is the shared scheduler (prefill/decode interleaving, shared prefix blocks, no whole-turn TTFT queueing), **not** one fused decode forward; the fused branch is currently dead code there. Gemma4 fuses ordinary `[N,1]` rows through full and sliding paged groups, including KV-shared anchors and PLE. Eligible all-greedy rows stay batched through one argmax/eval epilogue; mixed sampling or penalties retain scalar per-row semantics. `schedulerStats().fusedGreedyEpilogueSteps` counts executor-confirmed fused epilogues rather than inferring them from planned occupancy. Every request retains private penalties, sampling, stop, stream, and cancellation state. |
| Transport     | SSE writes honor Node backpressure and stop native pulls until drain or close. The 30-second default drain deadline aborts a connected stalled peer and releases admission. The chat model-thread output mailbox, native callback queue, and JS callback queue are independently capped at 64 events.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

Per-request prefix verification remains all-or-nothing. A miss releases and
rebuilds only that sequence's slot; it never resets a peer. Batched admission
uses fresh JS sessions while retaining the native refcounted prefix cache, so
correctness is independent of the warm single-slot registry.

## Scheduler knobs

The stable capacity knobs are read once per process. The ragged executor switch
is read once when a resident Qwen3 model-thread state is created, which permits
same-binary uniform/ragged validation with separate model instances:

| Knob                            | Default | Meaning                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLX_SCHED_MAX_NUM_SEQS`        | `8`     | Native running-set cap and server admission capacity (hard-clamped to 32).                                                                                                                                                                                                                                                                            |
| `MLX_SCHED_LONG_PREFILL_TOKENS` | `2048`  | Maximum prefill progress for one request in one step.                                                                                                                                                                                                                                                                                                 |
| `MLX_SCHED_WATERMARK_FRACTION`  | `0.05`  | Free-block headroom retained while work is already live.                                                                                                                                                                                                                                                                                              |
| `MLX_SCHED_RAGGED_STEP`         | `0`     | Use one packed varlen Qwen3 forward for mixed prefill/decode slices.                                                                                                                                                                                                                                                                                  |
| `MLX_LFM2_FUSED_BATCH_DECODE`   | `0`     | Opt LFM2 multi-row decode back into one fused `[N,1]` forward (default is a row-exact wave: per-row N=1 forwards, bit-identical to serial).                                                                                                                                                                                                           |
| `MLX_PAGED_PER_SEQ_CTX`         | `unset` | Optional per-sequence **admit** cap and Qwen3/LFM2 pool-budget clip. Unset → admit `min(trained, live pool minus recurrent KV-equivalent)`. When set, Muse/Nemotron `contextLimits()` also publish that cap. Default 32768 applies only to the Qwen3/LFM2 **pool request** formula. Qwen3.5 dense/MoE ignore it: their windows are the memory plan's. |
| `MLX_SERVE_FORCE_SERIAL`        | `0`     | Route eligible Qwen3/LFM2/Qwen3.5/Gemma4/Muse-Glimmer/NemotronH turns through the whole-turn path for A/B and rollback.                                                                                                                                                                                                                               |

`MLX_SERVE_FORCE_SERIAL` is a process-start test/rollback switch, not a
production hot-reconfiguration API. Set it before model registration; changing
it later can only fail safe because the server admission width was fixed when
the resident `SessionRegistry` was created.

Two reproducibility rules are deliberate:

1. Floating-point reduction differences can change greedy outputs across batch
   shapes. Ordinary AR sampling also depends on row-order PRNG consumption.
   Scheduled speculation owns each request's RNG; peers cannot consume its
   draws, though numerical changes can still alter the sampling distribution.
2. Each request's legal prefill break-set is pinned at admission. The shared
   2048-token step budget decides when a pinned slice runs, never where it is
   split; this preserves family-specific chunk-boundary invariants. Qwen3.5
   dense and MoE are the exception: they size prefill rows by the measured
   cut below and read no grid.
3. The scheduled lane treats `max_new_tokens` as an upper bound and clamps it
   to the remaining per-sequence context. Large OpenAI/Anthropic output hints
   therefore end at `length` instead of failing admission solely because the
   hint exceeds the pool window.

Admission remains FCFS. If the oldest request cannot yet satisfy the memory
watermark, smaller later requests wait behind it; a preempted row likewise
keeps its original queue position until it can resume. This deliberate
head-of-line blocking prevents a stream of small arrivals from starving a
large request or an ordered reset behind it. A `WaitingForSsd` row remains
charged because its reserved destination blocks and future-growth reservation
still consume the same unified-memory budget while I/O is parked.

### Prefill beside a decoding lane (Qwen3.5)

Qwen3.5 dense and MoE plan their prefill as Splash `Scheduler::planPrefill`.
Every other family keeps the fixed 2048-token step budget.

- **The unit.** A step's prefill rows together are one prefill unit, run
  after the step's decode rows. Decode rows (one token plus any draft rows)
  sit outside the unit. A unit holds at most four lanes.
- **Leader.** The ready lanes are sorted Splash `leadsBefore`: a lane passed
  over by three units that served later arrivals is overdue and leads
  (oldest overdue first); otherwise the fewest remaining prompt tokens lead,
  then arrival. Cancelled lanes and lanes whose memory retry is not due are
  left out, and are not passed over; a lane refused its row in a step is.
  A cancelled prefill fails in a one-token row after the unit.
- **Pair bound.** The leader's rows (at most 2048) halve, down to 64, while
  its attention pairs `rows·position + rows(rows+1)/2` exceed those of a
  2048-row chunk at the end of the native window.
- **The cut.** With no lane decoding the unit is 2048 rows. While a lane
  decodes it is `cutRows`: 512 rows until a unit has been timed, then the
  rows the last timed unit ran in G = 500 ms, scaled down by how many more
  pairs the leader reaches at its depth, clamped to [512, 2048]. A rest below
  512 rows runs with the cut, up to 2048.
- **Joiners.** Each other ready lane, in lead order, joins with
  `(remaining − 1) % 2048 + 1` rows (the fewest that spare it a unit of its
  own) when that fits under the unit and the pair bound.
- **Sample.** The unit's measure is its rows, pairs and the step's prefill
  phase wall, taken only when every planned row ran timed. A refused or
  cancelled row is one the unit ran without.
- **Decode debt.** A unit run while lanes decode owes them the decode share
  (0.5) of its wall; while debt is owed the step plans decode rows only, and
  their wall works it off. The debt drops when no lane decodes.
- **Admission.** A prepared request takes no memory until the unit it would
  join serves it: it ends its prompt there, or no lane ahead of it is left
  with its own prompt unfinished (Splash `prefillAdmissionOrder`). A live
  continuation is planned from the tokens it keeps; any other cached prefix
  counts as uncached until a prefix probe exists. A request held back waits
  for scheduling: its memory wait ends and its claim holds nothing.
- **G** is one constant, shared with the decode-gap classifier's 500 ms
  threshold.
- **Decode share.** `--decode-share` (`mlx serve`, `mlx agent`; load option
  `decodeShare`) sets the share, a finite number of at least 0; `0`
  alternates one unit with one decode step. Other families refuse it.
- **Status.** `/health` `memory_governor.scheduler` has one entry per loaded
  Qwen3.5 model (Splash `/status` `scheduler`, read live): `model` (the name
  it serves under), `model_id`, the gauges `queued`, `waiting_resources`,
  `waiting_prefix`, `prefilling`, `decoding` (each request by phase at the
  end of the model thread's last pass; a request not yet started is queued,
  or waits for resources once refused a lane or memory; a running lane
  waiting for its retry still counts as prefilling or decoding), the counters `prefill_batches`,
  `prefill_rows`, `prefill_units`, `bound_cuts` (units whose leader was cut),
  `decode_batches`, `decode_batches_by_width` (`b1`, `b2`, …),
  `decode_gap_target_ms` (G), `contended_prefill_ms` (`p50 p95 p99 max total
  samples` over the walls of units run while a lane decoded, 1 ms buckets)
  and `replans`. `memory_governor.decode_gap_causes.prefill_ms` sums, over
  the gaps past G that a unit caused, the prefill walls the lane waited
  through without a row in them.

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
recompute preemption   ✓ LIFO victim, no same-step readmit, hot-prefix resume
Qwen3/LFM2/Qwen3.5/NemotronH adapters  ✓ per-sequence request table + shared prefix blocks
Gemma4/Muse-Glimmer grouped adapters   ✓ full + sliding pools, null-block retirement
Qwen3/Qwen3.5 executors        ✓ one uniform [N,1] decode forward
LFM2 executor                  ~ row-exact wave by default (N single-row
                               forwards, bit-identical to serial); fused [N,1]
                               behind MLX_LFM2_FUSED_BATCH_DECODE=1
Gemma4 executor              ✓ one fused [N,1] hybrid decode forward
NemotronH executor           ~ fused [N,1] on a DENSE checkpoint only; any
                               quantized one (the only published checkpoint)
                               runs N serial single-row decodes instead
LFM2 recurrent state   ✓ private [l_cache-1, hidden] row per conv layer/request
Qwen3.5 GDN state      ✓ private conv + recurrent row per linear layer/request
NemotronH mamba state  ✓ private conv [3,6144] + SSM [64,64,128] f32 row per SSM layer/request
NemotronH prefill      ~ executed slices are re-split on the config chunk grid;
                         the slice's own start/end still follow the engine's
                         token-budget grid and can land mid-chunk
BlockAllocator         ✓ refcounts + prefix hash (vLLM-style pool)
FFI / Metal kernels    ✓ num_seqs = q.shape(0), grid.y = sequence
ragged mixed step      ✓ Qwen3 env-gated SEAM B executor swap; scheduler unchanged
Gemma4 owner routing         ✓ paged AR + fixed/greedy-adaptive DSpark; media/assistant exclusive
```

- Kernels/FFI: `crates/mlx-paged-attn/metal/attention/paged_attention.metal:762-806`,
  `crates/mlx-sys/src/mlx_paged_ops.cpp:915`, launch grids in
  `crates/mlx-paged-attn/src/metal/paged_attention.rs:1112-1373`.
- Allocator: `crates/mlx-paged-attn/src/block_allocator.rs:67-129` (refcounted
  `PhysicalBlock`, `prefix_cache`, `find_longest_cache_hit`, `cache_full_blocks`);
  per-model-instance, not process-global.
- The adapter request table and batched metadata are in
  `crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs`; the dense Qwen3
  executor is in `models/qwen3/model.rs`, LFM2's ShortConv executor hooks are
  in `models/lfm2/model.rs`, Gemma4's grouped-cache executor is in
  `models/gemma4/scheduler.rs`, and their common admission, owner, SSD-wait,
  preemption, completion, and barrier lifecycle is in
  `engine/hybrid_scheduler.rs`. Every family implements only
  `HybridSchedulerBackend` cache/state hooks beside its tensor program.

Remaining scaffolding:

| Scaffolding                                                      | State                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SequenceBlockTable::replace_block` ("for copy-on-write")        | test-only callers (`block_table.rs:63`)                                                                                                                                                                                                                      |
| Varlen kernel (`cu_seqlens_q`, N ragged sequences)               | Qwen3 mixed scheduler steps and family-specific single-request paths; Qwen3 builds one physical-table row and one genuine cumulative-query boundary per request                                                                                              |
| Duplicated-row prefill layout (per-row heterogeneous `seq_lens`) | env-gated escape hatches only: gemma4 `MLX_GEMMA4_PAGED_PREFILL_ROUTE=legacy`, lfm2 `MLX_LFM2_PAGED_PREFILL_PAGED_ATTENTION=1`; qwen3.5's scheduled verify uses the varlen bridge; its serial prefill and verify attend in place (`paged_prefill_attention`) |

The tuned grouped D128/D256/D512 long-context kernels remain gated to
`num_seqs == 1` (the kernel indexes one sequence per dispatch); dense Qwen3 head
dimension 128 uses the generic batched route. A consequence: a scheduled row's
attention bits depend on co-scheduling. Muse-Glimmer's global layers take the
grouped D128 kernel when a row decodes alone (in the scheduled or the
whole-turn lane) and generic V2 when other rows share its step, so the same
greedy request can give a different transcript under load. Gemma4 now behaves
the same way on its D512 layers: a one-row scheduled wave forwards through the
whole-turn single-token step, so the row resolves the same decode plan and
grouped kernel either lane would pick, and a multi-row wave still takes
generic V2 — parity is exact whenever a row decodes alone, and prefill walks
one shared slice grid (`prefill_slice_ends`) regardless of which lane admitted,
restored, or replayed the prompt (see
[splash-qwen38.md](research/splash-qwen38.md) §7 item 12).

## Stage 0 concurrency hazards (verified status)

| #   | Hazard                                                         | Mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1  | Event-loop freeze after a streaming abort — **fixed**          | `resetCaches()` is async NAPI and waits on the model command channel without blocking Node. Streaming and synchronous chat turns install a turn cancel flag; flat/paged prefill, MTP/hidden-prefill helpers, and GDN materialized replay poll it at chunk boundaries and fail closed without publishing partial cache state. Residual: a single-shot prefill remains atomic, but a queued reset still parks only its promise, never the event loop. (`models/chat_napi.rs:77`, `engine/backend.rs:868`, family `turn_cancel` checkpoints.)                                                                                                                                                                                                                                                                                                                             |
| H2  | Non-streaming requests cannot be cancelled — **fixed**         | Public LM calls keep their ordinary names and accept the platform-native `AbortSignal`; `ChatSession` send paths pass `opts.signal`. Internally, the wrapper bridges to a two-phase native operation with one shared atomic flag. All supported session models, including Qianfan-OCR, poll it at safepoints. Both HTTP endpoints abort on disconnect and skip dispatch when the peer is already dead. Cancelled turns reject exactly with `"chat session cancelled"` and roll JS history back.                                                                                                                                                                                                                                                                                                                                                                        |
| H3  | Unbounded queue by default — **fixed**                         | `createServer` defaults the per-model cap to 16 waiters; env/config/host options can override or explicitly select `'unbounded'`. The coordinator bounds arrivals during cold load before a `SessionRegistry` exists. After resolution, pre-dispatch permits and FIFO waiters share the same atomic budget. Over-cap requests get 429 + `Retry-After: 1` with separate queue/pre-dispatch diagnostics.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| H4  | SSE ignores backpressure — **fixed with bounded cancellation** | Endpoints stop pulling when `res.write()` returns false and wait close-safely for drain. A 30s default drain deadline converts a connected stalled peer into a sticky abort and destroys the transport, so an admission slot cannot be held forever. The chat model-thread mailbox, native TSFN delivery, and JS callback queue each have a 64-event ceiling. A full mailbox backpressures the model instance's one producer OS thread, shared by every row in its active batch; close or callback overflow drops the receiver, wakes that producer, and cancels the turn. A slow-but-continuously-draining row can therefore throttle its peers until that turn ends because it never trips an overflow or the stalled-peer deadline. Per-row output parking remains a performance follow-up. Qwen3-ASR's real-time capture stream is outside this chat-output claim. |

Cross-model residual coupling is perf-class, not correctness: the process-wide
Metal wired limit is a model-life lease for qwen3_5 / qwen3_5_moe / qwen4_exp
(`WiredResidency` in `crates/mlx-core/src/stream.rs`, lapsing after the
`--idle-release` time without a command) and is set/restored per turn for the
other families; the process limit is the largest live lease,
and the flat decode path calls `clear_cache` every 256 steps, draining the
process-wide Metal free pool (`ChatBackend::maintain_cache`; noted as
multi-model-hostile on `handle_chat_cmd` in `engine/cmd.rs`).

## Yardstick: vLLM v1

The vLLM reference stack has since moved to the Model Runner V2 default, where drafts run as
ordinary scheduled tokens — ours stay a barrier turn (see
[vllm-speculative-alignment.md](vllm-speculative-alignment.md), X10); the rows below were verified
against v1.

| Mechanism                                                                   | vLLM v1                                                                                                                               | mlx-node today                                                                                                                                                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Continuous batching (one shared forward per step; no prefill/decode phases) | Core scheduler loop                                                                                                                   | Qwen3, LFM2, Qwen3.5 dense/MoE, Gemma4, Muse-Glimmer, and NemotronH text rows share one scheduler and standard step executor; only Qwen3's optional ragged tensor packer supplies a specialized executor |
| Per-step token budget + chunked prefill                                     | Enabled by default                                                                                                                    | One 2048-token ceiling with decode-first planning and pinned per-request breaks                                                                                                                          |
| Live prefix sharing                                                         | Chained hashes, refcounts, and LRU                                                                                                    | Refcounted block pools across supported paged families; Qwen3, Qwen3.5 dense/MoE, LFM2/2.5, Gemma4, and Muse-Glimmer restore complete model state from SSD                                               |
| Admission / preemption                                                      | `max_num_seqs`, watermark, preempt-and-recompute (freed blocks stay prefix-cache-reusable; optional CPU/secondary offload connectors) | Sequence cap, reserve-aware watermark, and allocation-squeeze LIFO recompute; measured long prefixes may capture to SSD                                                                                  |
| Backend seam                                                                | Scheduler drives platform-specific workers                                                                                            | `HybridSchedulerBackend` gives every supported family one request lifecycle and standard step executor while family hooks retain architecture-specific tensor/cache operations                           |

What does not transfer: CUDA-graph padding (MLX lazy graphs play that role),
the multi-process ZMQ split (a scheduler thread suffices at this scale),
and "use all GPU memory for KV" pool sizing (on unified memory the pool
competes with the weights — see `docs/architecture.md`). Deeper async
scheduling needs explicit in-flight ownership and completion tracking. Its
benefit must be measured; shared memory does not eliminate GPU completion waits.

### Architecture follow-up after the vLLM comparison

Four additional mechanisms fit the single-process MLX runtime:

- The chat request-output path is bounded end to end. `StreamTx` backpressures
  the model instance's single producer OS thread, while the Tokio forwarding
  pump remains runnable and can observe close/cancellation. Because every row
  in a scheduled batch shares that producer, a slow-but-draining consumer can
  throttle peer rows; per-row output parking is a performance follow-up.
- The common deterministic decode case keeps `[batch, 1, vocab]` logits intact
  through one last-axis argmax and one device evaluation. Mixed waves build
  per-row samplers in order, including penalties and forced reasoning tokens,
  then evaluate their token/logit roots together. Random draw construction order
  stays unchanged; a row preparation failure remains local to that row.
- `ChatConfig.cacheSalt` and the Responses/Anthropic `cache_salt` fields define
  an explicit prefix-cache security domain. The SHA-256-derived compact domain
  is threaded unchanged through lookup and publication and is independent from
  `prompt_cache_key`/session ownership.
- `KVCacheCoordinator` owns one runtime manager aligned with every compatible
  `KVCacheGroup` and the authoritative per-layer routes. Gemma4 consumes those
  routes directly: full and sliding groups own separate physical pools;
  KV-shared layers alias their anchor; sliding eviction preserves logical table
  width with a null block. SSD restore reconciles the full and every sliding group
  at one validated boundary. Cross-owner in-process hot reuse remains disabled;
  the persisted sidecar is the joint commit record a hot full-group hit lacks.

Storage correctness does not imply that every family has the same scheduling
throughput. Qwen3.5 MoE now defaults to paged K/V, has exact GDN-backed SSD
restart parity, and admits eligible plain-text AR through the hybrid scheduler by default
(up to 4 lanes, computed from the machine's memory at load; up to 8 on machines with 256 GB or more). Fixed-depth native MTP also uses this lane; adaptive MTP,
DFlash2 and media remain whole-turn exclusive. Hybrid sidecar
restores also remain synchronous on the model thread, and Gemma4 still declines
cross-owner in-process hot hits. These are latency/occupancy follow-ups, not
holes in the paged or durable-state contract.

## Validation evidence and performance boundary

The real-checkpoint correctness gates run independently of the benchmark:

- Qwen3-0.6B BF16: serial/uniform/ragged results are token-identical, uniform
  and ragged streaming terminals are byte-identical, mixed per-row penalties
  remain isolated, a real scheduled wave reaches occupancy 8, and the
  penalty-free N=2 gate requires the fused-greedy engagement counter to
  advance rather than accepting scalar-fallback parity.
- LFM2-1.2B BF16: the recurrent-state batched parity gate passes.
- Qwen3.5-0.8B BF16: the scheduler-driven asymmetric-finish and cross-owner
  warm-wave gate passes with occupancy 2, including on a checkpoint that has
  installed vision and MTP modules while the tested turns explicitly select
  plain text AR.
- Qwen3.5 MoE: a synthetic Metal fixture with genuine routed/shared experts
  proves one N=2 paged/GDN forward returns the same two greedy tokens as scalar
  replay. A local `Qwen3.6-35B-A3B-mxfp4-mlx` gate additionally matches serial
  output through asymmetric completion and a cross-owner warm wave while
  recording occupancy 2. The smallest published checkpoint is 35B-A3B, so
  standard CI does not claim a real-checkpoint throughput result; the tiny
  correctness fixture measured slower than serial and is not a performance
  gate.
- Gemma-4-E2B-IT: serial and concurrent starts plus continuations are
  byte-identical at T=0, and native scheduler telemetry records a genuine
  occupancy-2 fused decode wave. A direct random-weight N=2 hybrid forward also
  matches the two serial argmax rows through full/sliding/KV-shared routing.
- The real HTTP Stage-1 server gate observes two simultaneously active SSE
  handlers and native decode occupancy 2. Each `ChatSession` supplies a stable,
  request-local cache owner; without that identity native correctly falls back
  to the legacy exclusive sequence-zero lane.

`ChatSession` pins its first effective `cacheOwnerId` for the wrapper's entire
lifetime; a later per-call attempt to switch identities rejects before native
dispatch. `reset()` releases only that owner on block-paged models, preserving
unrelated live sessions, while exclusive/flat models retain their model-wide
reset barrier. `dispose()` releases the same stable owner. Do not share one
explicit owner id between independent sessions: resetting or disposing either
wrapper releases the shared native scheduler state. Server warm-slot eviction
and stateless request cleanup await this lifecycle transition before dispatching
replacement work. Native also rejects a second in-flight turn for the same
owner before any family-specific start reset can replace its cache. If paged
admission is waiting on memory held by completed owners, idle-owner release
commands may bypass ordinary pending work. Global resets and ordinary inference
turns remain FIFO and cannot use this cleanup lane to overtake earlier work.

Performance evidence is deliberately reported separately. On 2026-08-10, an
Apple M5 Max with 128 GiB ran one uncooled fresh-process A/B sample against
Qwen3-8B BF16 (4,096-token prompts, 512-token outputs). The scheduled worker
reached occupancy 1/2/4/8 and measured aggregate speedups of 1.019x, 1.803x,
2.786x, and 3.723x. Mixed-wave chatter p95 TTFT improved from 73.824 s to
14.423 s. This is mechanism/directional evidence, not a claimed ship-gate
pass: it was one run with zero cooldown, N=1 server TTFT was 1.587x the serial
measurement, and the N=4/N=8 values are below the draft 3.0x/4.5x thresholds.

The draft's qualifying checkpoint premise was also invalid: dense Qwen3
currently rejects quantized weights, so a "dense Qwen3 8B 4-bit" run cannot be
performed by this runtime. The loader now accepts both single-file and sharded
dense safetensors checkpoints, but the 4-bit thresholds remain non-applicable
until plain-Qwen3 quantized execution exists. Do not represent the BF16 sample
above as a cooled median-of-three 4-bit result.

## Direction

- **Stage 0 — robustness (landed):** H1–H4 are regression-locked: ordinary
  non-streaming APIs with `AbortSignal`, pre-dispatch disconnect checks,
  fail-closed chunk-boundary prefill cancellation, async reset, cold-load and
  resident admission caps, and deadline-bounded SSE backpressure.
- **Stage 1 — dense Qwen3 continuous batching (landed):** the phase-free
  scheduler, uniform batched decode, per-row epilogue, live prefix sharing,
  block-watermark admission, asynchronous SSD restore, server semaphore, and
  same-binary forced-serial rollback path are implemented.
  Token parity is the correctness gate; the occupancy histogram is executor
  self-reporting and therefore not, by itself, proof that a fused forward ran.
  The fresh-process wall-time ship gate is the independent non-vacuity check.
- **Stage 1.5 — LFM2 hybrid entry (landed):** each live request owns a
  per-conv-layer `[l_cache-1, hidden]` recurrent-state row. Decode stacks those
  rows, runs each ShortConv once over `[N,1,H]`, and scatters the next state;
  cached-prefix Pass 1 runs at most once per admission.
- **Stage 2a — preemption (landed):** a genuine paged allocation squeeze pops
  the newest running victim, publishes verified hot hashes, releases its live
  table, and prepends it for replay without same-step readmission. Recompute is
  the default; only a long prefix with measured restore throughput that beats
  measured prefill cost escalates to asynchronous SSD capture. Public scheduler
  stats expose total, recompute, and SSD preemption counters. The comparator is
  intentionally conservative under allocation pressure; it is a tuning policy,
  and measured restore/prefill telemetry—not a semantic guarantee—decides the
  cheaper path.
- **Stage 2b — ragged Qwen3 (env-gated):** `MLX_SCHED_RAGGED_STEP=1` packs
  decode rows and prefill slices into one token stream, writes K/V once per
  layer, and passes genuine cumulative query boundaries to varlen attention.
  Logits are selected at `query_start_loc[1:] - 1`. The scheduler is unchanged;
  the random-weight mixed-step oracle proves a real one-forward decode+prefill
  path, though its tiny model is not a throughput gate.
- **Stage 2c — wider hybrid execution (landed):**
  text-only Qwen3.5 dense and MoE store up to 4 request-local GDN units (the
  lanes their memory plan grants; up to 8 on machines with 256 GB or more),
  fuse
  their `[N,1,H]` decode, and reconcile preemption through the deepest
  K/V-backed GDN checkpoint/sidecar boundary. Paged-block growth and recurrent
  state debit one byte budget. Eligible text-only paged checkpoints take this
  lane by default (`MLX_SERVE_FORCE_SERIAL=1` rolls back to whole-turn); the
  memory plan prices the MTP state once, beside the pool, either way.
  Fixed-depth native MTP now shares per-owner
  verification; adaptive MTP, DFlash2 and media retain exclusive lanes. The tiny
  random-weight dense and genuinely sparse MoE fixtures prove token identity
  but are slower than two
  scalar forwards, so no real-checkpoint throughput win is claimed.
  The B12 routing premise was re-derived against `gated_delta`: `Auto` uses
  per-step GDN on every architecture, while chunked GDN is explicitly forced
  and is ineligible for this lane. Scheduled rows therefore always use the
  per-step recurrent-state representation described above.
  NemotronH joins the scheduled lane default-on: its six GQA layers use the
  generic batched paged route (32/2 heads, head_dim 128) and per-request
  Mamba-2 rows are stacked/scattered around each SSM layer for one `[N,1]`
  decode forward — on a dense checkpoint. Quantized checkpoints (the only
  published one included) take the `row_exact_decode_projections` branch
  instead: N serial single-row decodes, bit-identical to serial, no fused
  decode forward. Prefill chunk alignment is best-effort, not an invariant:
  the engine's break-set is a plain `MLX_SCHED_LONG_PREFILL_TOKENS` (2048)
  grid walked from the effective cached-prefix length
  (`engine/hybrid_scheduler.rs:1772-1793`), NemotronH overrides neither
  `scheduler_prefill_slice_tokens` nor `extra_prefill_breaks`, and the
  per-step planner budgets decode rows first, so a step with D concurrent
  decode rows hands the prefill at most `budget - D` tokens
  (`engine/scheduler.rs:1123-1140`). `run_scheduled_prefill_slice`
  (`models/nemotron_h/model.rs`) re-splits whatever range it is
  handed with `chunk_aligned_prefill_slices` on `config.chunk_size` (128 on
  the released checkpoint — read from config, never hardcoded), so every
  boundary _interior to one scheduled slice_ is a chunk multiple; the
  slice's own start and end are not, and can land mid-chunk. That is safe
  rather than wrong: the chunk scan pads the final chunk of each forward
  (the `pad_size` step in `models/nemotron_h/mamba2.rs`) and carries the recurrent state
  across, so a mid-chunk boundary changes the reduction order, not the
  semantics. A **synchronous** MTP turn stays exclusive and speculates: the
  plan resolves it to the FLAT speculative core (`TurnPlan::resolve`,
  `engine/plan.rs`), because the family declares
  `supports_paged_attention: false` and a flat-only decoder takes the flat
  lane WITH its target. A **streaming** MTP turn is a different shape: the
  flat core has no streaming arm (`supports_streaming: false`), so the plan
  resolves it to plain paged AR and `chat_requires_barrier` leaves it on the
  scheduled lane. The barrier gate and the planner read one answer
  (`SpeculativePlan::admits_streaming`); neither family code nor the
  scheduler holds a second opinion. Symbols, not line numbers: this file moves
  every week. That barrier is exactly why the family does NOT auto-enable
  MTP: `mtpAutoEnabled()` returns false unless `MLX_NEMOTRON_MTP_DEFAULT=1`,
  so the scheduled lane is what a default session gets. The old throughput
  justification for that default (a 0.56x MTP-vs-AR measurement) is
  retracted — post-fix, MTP measures as a wash (1.00-1.04x vs flat AR,
  0.994x vs paged AR), and the surviving reason is this loss of batching,
  not a loss of tokens/s. See docs/models.md.
  The tiny random-weight fixture proves N=2 batched==serial T=0 identity in
  CI on every run. All **three** of the family's real-checkpoint gates —
  `nemotron_h_paged_vs_flat_parity`, `nemotron_h_concurrent_batched_parity`,
  and `nemotron_h_mtp_midcycle_state` — are listed on the `nemotron_h` leg of
  the `model-e2e` matrix, alongside the lib-level `real_mtp_t0_lossless_gate`.
  Listed is not the same as run. That leg has **never executed**: the whole
  `model-test` job is gated on the `model-e2e` PR label, and this leg's run
  step additionally exits early on `push` and `workflow_dispatch`, so no
  release path touches it. It also asks for more machine than a hosted runner
  has — ~38 GB of free disk for the 30B source plus its converted output, and
  the paged-vs-flat gate loads a flat clone _and_ a paged clone of the ~21 GB
  model and holds both resident for the whole comparison (~42 GB). Treat local
  runs as the family's only real evidence until that leg has gone green once on
  hardware that fits it.
  Gemma4 replaces the former per-row rotating-cache no-go with a vLLM-style
  hybrid coordinator: full and sliding groups have paged storage, expired
  sliding blocks become null sentinels, KV-shared layers alias physical anchors,
  and ordinary rows execute one fused `[N,1]` forward. Dynamic recompute
  preemption shares the full-group pool instead of statically partitioning one
  maximum context per request. Fixed-depth text DSpark adds packed ragged target
  verification and independent accepted-prefix commits. Greedy adaptive DSpark
  measures per-owner allocation costs; sampled adaptive, media and assistant
  commands use ordered owner lanes. Reset and stats remain
  barriers. See the [inference reference](research/inference-2026-09-05/README.md#measurements-and-decisions)
  for throughput, numerical limits and recurrent MTP validation.
