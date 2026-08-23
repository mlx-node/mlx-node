# Block-paged KV cache (vLLM-aligned)

A vLLM-style block-paged KV cache lives alongside the legacy flat `Vec<KVCache>` path. Multiple in-flight requests share refcounted KV blocks for any prompt prefix they have in common (system prompt, shared few-shot preamble, repeated tool-result frames, etc.).

Routing is per-model via the `use_block_paged_cache: Option<bool>` config field.
Dense models use one paged group. Hybrid models may retain dedicated recurrent
state (LFM2 ShortConv and Qwen3.5 GDN), while Gemma4 maps both full and sliding
attention to separate paged groups and aliases KV-shared layers to their anchor.

## Foundation types

| Type                  | Location                                                    | Role                                                                                                                                                                                                |
| --------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BlockAllocator`      | `crates/mlx-paged-attn/src/block_allocator.rs`              | Logical lifecycle — per-block refcounts, LRU eviction, prefix-hash table for cross-request reuse                                                                                                    |
| `LayerKVPool`         | `crates/mlx-paged-attn/src/layer_kv_pool.rs`                | Physical storage — per-layer Metal K and V `Buffer` pairs sized to `paged_cache_memory_mb`                                                                                                          |
| `PagedKVCacheAdapter` | `crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs` | Session-friendly wrapper. Per-request lifecycle: `reset_for_new_request` → `find_cached_prefix` → `allocate_suffix_blocks` → `record_tokens` → `register_full_blocks_for_reuse` → `release_request` |
| `KVCacheCoordinator`  | `crates/mlx-core/src/transformer/kv_cache_spec.rs`          | Keeps declared cache groups, per-layer physical routes, and one runtime manager per group aligned. Gemma4 owns independent full/sliding paged managers through it.                                  |

`BlockAllocator` and `LayerKVPool` are intentionally split so the legacy `CacheEngineManager` path (used by `use_paged_attention`, a different flag — see below) is unaffected. `paged_cache_memory_mb` defaults to 2048 when `None`.

## Pool growth: start small, double on exhaustion

The `qwen3_5` and `qwen3_5_moe` loaders size the paged pool in two steps. The
**initial** pool is allocated at load from `paged_cache_initial_memory_mb`
(config.json) or `MLX_PAGED_CACHE_INITIAL_MB` (env wins over the config field;
a set-but-unparseable env value falls back to config). The **max** is the
existing budget — `paged_cache_memory_mb`, or the auto one-full-context default
when unset — minus whatever `load_time_pool_sizing` clamps for live unified
memory.

When a reservation outruns free + evictable blocks, the adapter grows before
evicting LRU cache-only entries (`PagedKVCacheAdapter::try_grow_pool`): the new
block count is `min(max, max(2 × current, current + needed))` — double the
current pool, or jump straight to what the reservation needs when that is
larger, never past the max. Each reservation gets one grow before allocation
plus one retry on mid-loop exhaustion; a reservation that still cannot fit
falls through to the existing eviction/error path. The load log reports both
sides distinctly — `initial_blocks=` plus `effective_window_tokens=` computed
from the max — and a dynamic pool adds a
`paged pool is dynamic (grow-on-demand)` line.

| Knob                              | Source                                  | Role                                                                                                                     |
| --------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `paged_cache_initial_memory_mb`   | config.json (qwen3_5 / qwen3_5_moe)     | Initial pool size in MiB. Unset = the max itself — the historical fixed-size pool, byte-identical behavior.              |
| `MLX_PAGED_CACHE_INITIAL_MB`      | env (wins over the config field)        | Same, u32 MiB. Unparseable values are ignored (config field, then the unset default).                                    |
| `paged_cache_memory_mb`           | config.json                             | MAX pool size in MiB (auto one-full-context when unset).                                                                  |
| `MLX_PAGED_CACHE_MEMORY_MB`       | env (agent override manager only)       | Floor the manager writes for the max into the cloned config; unparseable falls back to the 16 GiB floor.                 |

The agent override manager (`packages/lm/src/models/paged-config-override.ts`)
writes `paged_cache_initial_memory_mb = 2048` into the cloned config for
`qwen3_5` / `qwen3_5_moe`, overridable with `MLX_PAGED_CACHE_INITIAL_MB`, and
clamped to the resolved max so it never exceeds the pool ceiling. Other
families are not given the field — their loaders have no initial knob and must
not receive one. Checkpoints resolved WITHOUT the manager (a library caller
loading the path directly) keep the historical behavior: unset initial =
static full-size pool.

**Transient double memory during grow.** `LayerKVPool::grow_to` allocates the
new generation's per-layer Metal buffers wholesale and swaps them in; the old
buffers stay alive until outstanding GPU work and any cached MLX array views
drop their handles, so peak unified-memory use briefly approaches old + new
before the old generation is released. The growth notifier
(`CacheLimitCoordinator::update_pool`) re-debits the coordinator with the new
total at the same point, so the global cap tracks the grown pool rather than
the initial one.

## Prefix-cache security domains

The native `ChatConfig.cacheSalt` field (Responses and Anthropic APIs:
`cache_salt`) separates content-addressed KV reuse between security domains.
The runtime hashes the caller-provided string with SHA-256 into the allocator's
compact domain id and uses the same id for both prefix lookup and block
publication. The server limits the source value to 256 UTF-8 bytes. Omitting it retains the shared default domain for trusted
single-tenant use.

This is intentionally different from `prompt_cache_key` and `cacheOwnerId`:
those select a warm conversation/session, while `cacheSalt` controls whether
identical token blocks may be physically shared at all. Multi-tenant servers
should derive a stable, high-entropy value from authenticated tenant identity
rather than accepting an arbitrary client-selected namespace.

## Per-model support matrix

| Model             | Default | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------- | :-----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qwen3**         | **on**  | Greedy + prefix-reuse byte-equal vs. flat path on Qwen3-0.6B BF16. Opt out via `use_block_paged_cache: Some(false)`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **LFM2.5**        | **on**  | Same parity result on LFM2.5-1.2B. Hybrid arch — only `full_attention` layers go through the adapter; conv layers stay on `Lfm2LayerCache::Conv`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Gemma4**        | **on**  | Serial/concurrent start and continuation parity on Gemma-4-E2B-IT with a real occupancy-2 wave. Full and sliding layers use distinct paged groups; expired sliding blocks are replaced by a null sentinel; KV-shared layers alias their global/sliding anchor. Same-owner continuation is live; cross-owner prefix hits currently fail closed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Qwen3.5 Dense** | **on**  | Single-turn greedy parity and paged construction are verified on Qwen3.5-0.8B BF16. Full-attention K/V is paged; GDN recurrent state remains request-local and is checkpointed beside K/V for SSD restore. Explicit false and the environment override retain a rollback path.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Qwen3.5 MoE**   | **on**  | Uses the same paged full-attention, exact-boundary GDN sidecar, and two-row text scheduler contract as dense. Sparse expert routing stays batched over `[N,1,H]`; checkpoint-specific projections preserve exact greedy parity while K/V gather/attention remains batched. MTP/media turns remain exclusive. Real `Qwen3.6-35B-A3B-mxfp4-mlx` decode and SSD-restart parity gates remain local-only because no small published checkpoint fits the standard CI runner; synthetic Metal gates cover paged construction and N=2 token parity in CI.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **NemotronH**     | **on**  | Hybrid Mamba-2 SSM + GQA + MoE-FFN: only the six `full_attention` layers route through the adapter (32 Q / 2 KV heads, head_dim 128 — the generic batched paged route, no per-family kernel). Mamba recurrent state stays request-local (per-request `[3,6144]` conv + `[64,64,128]` SSM f32 rows per SSM layer, stacked/scattered around each batched decode) and is checkpointed only in-process (no SSD sidecar yet — the family is NOT cold-restore eligible). The scheduled lane is default-on, but the fused `[N,1]` decode is reachable only on a dense/bf16 checkpoint: `load_inner` (`persistence.rs`) sets `row_exact_decode_projections` for any quantized checkpoint — including the only published one — so the decode wave dispatches to `run_row_exact_decode_wave` (`model.rs`), N serial single-row decodes (bit-identical to serial, but N full weight streams), and the fused branch is dead code there. `run_scheduled_prefill_slice` re-splits each scheduled range on `config.chunk_size` (128 on the released checkpoint), so boundaries _inside_ one scheduled slice are chunk multiples; the slice's own start/end come from the engine's plain 2048-token grid minus that step's decode rows and can land mid-chunk, which the chunk scan absorbs by padding the final chunk (reduction order changes, semantics do not). A **synchronous** MTP turn stays on the exclusive ordered lane and runs the FLAT speculative core: the family declares `supports_paged_attention: false`, and `TurnPlan::resolve` gives a flat-only decoder the flat lane WITH its target rather than dropping it to paged AR. A **streaming** MTP turn has no flat arm to reach (`supports_streaming: false`, `model.rs`), so it plans as plain paged AR and keeps its scheduled slot. Symbols, not line numbers, throughout this row: `models/nemotron_h/` is under active change. All **three** real-checkpoint gates are listed on the `nemotron_h` leg of the `model-e2e` matrix (see the parity-gate table below), but that leg has never executed: the `model-test` job is gated on the `model-e2e` PR label and this leg's run step additionally exits early on push/dispatch, and its bill — ~38 GB of disk for source + converted output, plus ~42 GB resident for the two model copies `nemotron_h_paged_vs_flat_parity` holds at once — exceeds a hosted runner, so local runs remain the only evidence; synthetic Metal gates cover paged construction, N=2 batched==serial T=0 parity, and the break-set boundary rule in CI on every run. |

### Gemma4 hybrid groups and draft coexistence

Gemma4 follows vLLM's `HybridKVCacheCoordinator` shape rather than allocating
one maximum-context cache per request:

```text
layer specs ──group──▶ full pool ───────▶ GlobalPaged
                  └─▶ sliding pool ────▶ SlidingPaged(window)
KV-shared layer ───────────────────────▶ alias(anchor group, physical slot)
```

Each live owner has one sequence id across every group. The scheduler only
advances a row after every group accepts the same cursor, rolls all groups back
on a partial failure, and preempts the newest victim by releasing all of its
tables before recomputing. Sliding adapters retain absolute logical block-table
width: once a block is wholly outside the attention window its physical block
is freed and that logical entry becomes a shared null sentinel. This bounds
sliding residency without renumbering RoPE positions.

The Gemma4 KV budget is a shared pool budget, not
`max_context × max_num_seqs`. The automatic minimum holds one maximum-context
request in every group plus a null block and at least one starter block per live
row. Short rows can therefore batch up to the scheduler width; as aggregate
contexts grow, recompute preemption—not a static per-request partition—arbitrates
the shared blocks. The server exposes this admission width through
`maxConcurrentSequences()`.

MTP/DSpark and assistant drafts remain flat-cache algorithms, but loading one
does not turn the paged coordinator off. The model scheduler installs a
request-local flat owner lane only while that speculative command executes;
ordinary owners continue to use the resident grouped pools. A live owner may
not switch cache layouts mid-history, and releasing the owner drops both lane
registries plus metadata.

For Qwen3.5 (dense + MoE) both the flat and paged decode paths are pure-Rust
eager — the compiled C++ forward and its process-wide locks
(`DENSE_COMPILED_MUTEX` / `COMPILED_WEIGHTS_RWLOCK`) were deleted in the
chat-engine refactor (`ee88b92b`), so there is no compile state to corrupt and
no per-step lock is taken on either path. VLM checkpoints run their image
turns through the paged vision core when `paged_adapter` is present (plain AR,
MTP weights ignored), and fail loudly when it is `None` — the flat fallback is
text-only.

## SSD cold tier: hybrid families and the auxiliary sidecar

The cold tier (`crates/mlx-paged-attn/src/cold_cache.rs`) persists full paged K/V
blocks so a warm prefix survives a process restart. Whether a family may _restore_
from it is a correctness decision, gated by an allowlist that exists in two places
and is drift-guarded by a test:

| Side  | Symbol                                                            |
| ----- | ----------------------------------------------------------------- |
| Rust  | `COLD_RESTORE_FAMILIES` in `crates/mlx-core/src/cold_tier.rs`     |
| TS    | `COLD_TIER_RESTORE_FAMILIES` in `packages/agent/src/cold-tier.ts` |
| Guard | `packages/agent/__test__/cold-tier-families.test.ts`              |

Dense `qwen3` is sound because its pool covers **all** layers, so a restored block
reconstructs the whole prefix. Every supported hybrid family now pairs K/V with an
exact-boundary sidecar or a second paged group:

| Family              | State outside the primary full-attention pool | Restore companion                                                   | Restore-eligible |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------- | :--------------: |
| `qwen3` (dense)     | none — pool covers every layer                | n/a                                                                 |     **yes**      |
| `gemma4`            | sliding-attention paged groups                | grouped live-window K/V sidecar; all groups install at one boundary |     **yes**      |
| `qwen3_5` (dense)   | GDN recurrent state                           | exact-boundary GDN sidecar                                          |     **yes**      |
| `qwen3_5_moe`       | GDN recurrent state (same as dense)           | exact-boundary GDN sidecar                                          |     **yes**      |
| `lfm2` / `lfm2_moe` | ShortConv recurrent state                     | exact-boundary `ConvState` sidecar                                  |     **yes**      |
| `nemotron_h`        | Mamba-2 SSM recurrent state                   | none yet (in-process only)                                          |      **no**      |

The allowlist is enforced **natively**, not only in the agent:
`cold_tier::resolve_persist_cold` consults `cold_restore_supported(model_type)`
before any other signal, so a family that is off the list never persists or restores
— not under an explicit `persist_paged_cache` config, not under
`MLX_PERSIST_PAGED_CACHE=1`, not via a direct library caller that bypasses the agent.
A loader may therefore carry a fully wired cold bracket ahead of proving it; the gate
keeps that bracket dormant until the family is admitted.

A K/V-only restore for a hybrid would resume from state the pool never held. Two
mechanisms make that impossible rather than merely unlikely. Gemma4 applies the
same rule to its physical groups: the full group proposes a boundary, a validated
sliding sidecar reconstructs every sliding group there, and any decode/install
failure resets all groups to zero.

Hybrid sidecar capture uses the same `cache_salt` as the K/V chain. The salt is
part of the first-block domain for both the hot prefix checkpoint and the
persisted GDN/ShortConv/sliding sidecar key, so a salted request can use the cold tier
without either crossing domains or failing at finalization.

**Reconcile-down (`ColdSidecarPolicy`).** A family whose out-of-pool state _is_
serializable attaches a `ColdTierContext` carrying a policy. The restore walk
(`ColdTierWalk::restore_extend`) then follows vLLM's per-group rule
(`vllm/v1/core/kv_cache_coordinator.py`, `sched/scheduler.py`): the K/V chain
proposes a candidate length, and the policy may only **reduce** it to the deepest
boundary a _validated_ `ColdSidecar` actually backs. Nothing backed means restore
nothing — never a "close enough" prefix. Sidecars live in their own filename
namespace under a group-tagged key (`ColdGroup`, vLLM's `BlockHashWithGroupId`),
with their own metadata and SHA-256 payload checksum; malformed input is a graceful
miss that prunes and counts a corruption.

**The hot-hit latch.** The walk only ever runs on what the in-process prefix cache
did _not_ already serve, so a block a backed restore published can come back later as
a pure hot hit with no sidecar attached. `PagedKVCacheAdapter::aux_prefix_unbacked`
latches that case, and `record_tokens` / `register_full_blocks_for_reuse*` /
`finalize_turn_keep_live*` all fail closed until the family calls
`confirm_aux_prefix_primed`. Families with no policy (dense `qwen3`) never latch.

### Writer durability: `fsync(2)`, not `F_FULLFSYNC`

The single writer thread commits every object with
`create_exclusive → write_all → sync_payload → renameat → directory fsync`.
`sync_payload` is deliberately **not** `File::sync_all`, which on Apple targets is
`fcntl(F_FULLFSYNC)` — a device-wide flush of the drive's own write cache. Because the
writer is one thread, every queued block waited behind that drive round trip.

Measured by `write_decomposition_bench::bench_write_path_phase_decomposition`
(`#[ignore]`d; release build, 128 rounds, round-robin over the sync variants, one cache
root per variant), per-object writer service time `Tw`:

| block                | `sync_all` `Tw` | `fsync(2)` `Tw` | speedup | of which device flush |
| -------------------- | --------------- | --------------- | ------- | --------------------- |
| qwen3_5 dense 198 KB | 4.267 ms        | 0.432 ms        | 9.88x   | 3.876 → 0.089 ms      |
| qwen3_5 MoE 330 KB   | 4.432 ms        | 0.446 ms        | 9.94x   | 4.007 → 0.098 ms      |
| qwen3-0.6b 1.84 MB   | 5.788 ms        | 1.273 ms        | 4.55x   | 4.773 → 0.236 ms      |

The flush is a fixed cost, so the smaller the block the more of `Tw` it was — which is
why the two blocks that actually ship gain roughly twice what the 1.84 MB bench block
does.

It also dominates the frontier `N = (Q+1)/(1 - Tc/Tw)`: the blocks
`ColdTierWalk::capture_chain` gets to persist before the bounded queue refuses one and
the walk breaks. `bench_chain_advance_per_turn` drives the real writer at a fixed
producer interval and counts the accepted prefix directly rather than deriving it, and
`bench_capture_cost_per_geometry` measures the producer cost `Tc` — one
`read_block_all_layers` round trip — at each geometry:

| block         | measured `Tc` |
| ------------- | ------------- |
| qwen3_5 dense | 0.215 ms      |
| qwen3_5 MoE   | 0.201 ms      |
| qwen3-0.6b    | 0.394 ms      |

Blocks accepted per turn at queue depth 8, two runs:

| dialled producer `Tc` | dense, `sync_all` | dense, `fsync(2)` | MoE, `sync_all` | MoE, `fsync(2)` |
| --------------------- | ----------------- | ----------------- | --------------- | --------------- |
| 0.10 ms               | 9                 | 11, 13            | 9               | 11, 10          |
| **0.20 ms (real)**    | 9                 | **28, 25**        | 9               | **17, 18**      |
| 0.32 ms               | 9                 | 600+, 403         | 9               | 72, 45          |

Under `sync_all` the answer is 9 at every producer rate — the writer was so much slower
than the producer that `N` collapsed onto `Q + 1` and nothing else mattered. That is
the ~8-9 blocks (~130 tokens) per turn seen in practice.

At the measured `Tc` the honest reading is the middle row: **9 → ~26 blocks per turn
(dense) and 9 → ~18 (MoE)**, i.e. 144 → ~410 and ~285 tokens. Reaching an 8192-token
boundary (512 blocks) drops from ~57 turns to ~20 (dense) and ~29 (MoE). `N` is steep
in `Tc`, so do not read the 0.32 ms row as the result — it is 50% past what capture
actually costs.

The two benches cross-check. Fed each geometry's measured `Tc`, the frontier formula
predicts `N` = 16.4 for MoE (observed 17-18) and 17.0 for dense (observed 25-28). MoE
lands on it; dense beats it, in the direction the setup predicts — the decomposition
bench interleaves all four sync variants round-robin on one device and `F_FULLFSYNC` is
device-wide, so a baseline write drags on the `fsync(2)` sample that follows it. Its
`fsync(2)` column is an over-estimate, which makes every speedup above the conservative
one.

**What is given up.** `fsync(2)` hands the bytes to the drive but does not ask it to
flush its volatile write cache, and drops the implicit device ordering barrier between
the payload extents and the journalled rename.

|                     | process kill | kernel panic | sudden power loss / hard reset |
| ------------------- | ------------ | ------------ | ------------------------------ |
| `F_FULLFSYNC` (was) | safe         | safe         | safe                           |
| `fsync(2)` (now)    | safe         | safe         | **at risk**                    |

This is affordable **only** because every read re-derives the SHA-256 payload checksum
recorded at write time (`decode_block`, `decode_sidecar`) and `load_object_bounded`
turns any decode error into a miss + prune + one `corruptions` count. After a hard
power cut a user can therefore see a slower-than-expected first turn, a non-zero
`corruptions` count on the dashboard, and a cold-tier directory that shrank — never
wrong KV handed to inference. `a_bit_flipped_block_is_a_miss_that_prunes_and_counts_corruption`
and `a_block_without_a_payload_checksum_is_refused` pin both halves of that.

The checksum stays SHA-256. Swapping it for a non-cryptographic hash would break the
on-disk format, and the bench says it is not worth it at the sizes that ship: on the
dense block `payload_checksum` is 0.061 ms of a 0.432 ms `Tw`, so making it _free_
(the harness prints that projection) would reach only 0.366 ms — worth ~3 more blocks
per turn, against invalidating every cache on disk.

### The per-turn capture budget

Everything above derives `N` — how far one turn's chain advances — from the writer.
That was the bug, not the analysis. `N` was **emergent**: it moved with the
filesystem, it was never chosen, and end to end it left the restored prefix at
**1.4-6.2% of the prompt** after a full session.

| prompt             | restored | share |
| ------------------ | -------- | ----- |
| 7781 (qwen3)       | 208      | 2.7%  |
| 8025 (qwen3_5_moe) | 496      | 6.2%  |
| 8025 (qwen3_5 27B) | 112      | 1.4%  |

Every turn showed `enqueued=12 queueDrops=1`: the walk stopped because the queue
refused, and it refused at the same place regardless of prompt length.

`capture_chain` now spends an explicit budget instead, and waits for a queue slot
rather than treating a full queue as a stop:

|                          | memory pinned | ratchet             | disk-independent | bounded tail                 |
| ------------------------ | ------------- | ------------------- | ---------------- | ---------------------------- |
| break-at-refusal, `Q`=8  | 8 blk         | 9-14, emergent      | no               | no (unbounded on a RAM disk) |
| break-at-refusal, `Q`=64 | 64 blk        | 92-126, emergent    | no               | no                           |
| **budget, `Q`=8**        | 8 blk         | **= budget, exact** | **yes**          | **yes**                      |

- `MLX_COLD_CAPTURE_BLOCKS_PER_TURN` (default **128** = 2048 tokens at block 16)
  bounds the steady-state ratchet.
- `MLX_COLD_CAPTURE_BUDGET_MS` (default **250**) bounds the turn tail.
- `DEFAULT_QUEUE_DEPTH` stays **8** and is now purely a host-memory bound
  (`Q x block_bytes` in flight). Raising it no longer buys reach.

The budget counts blocks this turn WROTE, not blocks it walked: a `contains` hit on
an already-persisted block is an in-memory index probe and does not spend depth, so a
long persisted prefix cannot stall the ratchet before it reaches the first block that
needs writing.

**Why it still breaks at a failed capture rather than skipping ahead.** Both
`kv_chain_upper_bound` and the restore loop stop at the first key that is absent, so
the chain's REACH is the index of the first hole under either policy — skipping buys
nothing on the turn that hits the hole, and costs one discarded Metal blit per skipped
block on the inference thread.

**The RAM-disk case is now answered rather than warned about.** On a
`MLX_COLD_CACHE_DIR` fast enough that the queue never pushes back, the old walk covered
the whole prompt in one turn: a 64 K-token first turn was 4096 blocks x ~0.2 ms =
~0.9 s of turn tail, uncounted and uncapped. The budget caps it at 128 blocks, the
deadline caps it at 250 ms, and `[MLX_TRACE] paged cold_capture_walk … stop= elapsed_ms=`
reports both per turn (`stop=deadline` also warns, since it means the device could not
absorb the configured budget).

**One hazard the deeper walk creates, fixed in the same change.** Every hybrid family
offers its state sidecar microseconds after the K/V walk returns, onto the same queue.
While the walk stopped _because_ the queue was full, a non-blocking sidecar offer was
guaranteed to lose — and a dropped sidecar is worse than a dropped block, because the
restore reconciles down to the deepest boundary a validated sidecar backs, so losing it
makes the turn's whole chain unusable. The families now use
`enqueue_sidecar_before(…, now + budget.max_walk)`.

### Gemma4 sliding-window sidecars

Gemma4 continuous batching stores full- and sliding-attention K/V in separate
paged groups. The scheduled text route snapshots the live K/V suffix from every
sliding group, persists it under `ColdGroup::SlidingWindow`, and reinstalls every
group at the full group's reconciled boundary. Retired logical positions become
null sentinels, so the restored block table preserves absolute positions without
allocating the expired prefix. Missing, malformed, or partially installable state
resets all groups to zero; reusing only the full group is forbidden.

The older rotating-cache representation remains active for the flat/exclusive
route used by media and MTP/DSpark commands. It shares the same sidecar layout
rules below, while the grouped route encodes/decodes layer-major paged K/V arrays
directly. It is therefore not an obsolete compatibility path: the two codecs cover
the two cache layouts that intentionally coexist on one loaded Gemma4 model.

`crates/mlx-core/src/models/gemma4/sliding_sidecar.rs` persists one
`RotatingKVCacheSnapshot` per **physical** sliding layer (`is_sliding_layer &&
!is_kv_shared_layer` — KV-shared aliases hold no state of their own), `keys` then
`values`, layer-major, bf16, via `to_uint16_native` / `from_bfloat16` so no f32 round
trip happens.

A rotating cache holds `min(offset, window)` tokens, so the payload length varies
below the window. That is a payload-format fact, **not** a hit rule, and the policy
keeps the two apart: gemma4 builds its policy with
`ColdSidecarPolicy::new_boundary_scaled(layout, 2)`, declaring the token axis of
`[1, kv_heads, window, head_dim]` as the one that follows the boundary. So
`expected_at(b)` stamps `dims[2] = min(b, window)` with `bytes_per_tensor` scaled to
match, and `sliding_sidecar::layout_at` derives the identical value on the capture
side. **Any** positive block-aligned boundary is representable — there is no window
floor, and no sub-window boundary is skipped. What the layout pins down is a shape
_rule_, not a byte count: group, dtype, tensor count and every non-declared dimension
are still frozen by the policy, and only the one declared axis (with
`bytes_per_tensor` following it) is a function of the boundary. Padding is still not
an escape hatch: `RotatingKVCache::restore_snapshot` hard-checks
`cached_tokens == min(offset, max_size)`, which is exactly what a sub-window payload
carries, so it is a first-class pre-wrap state rather than a pad or a truncation.

At and above the window the scaling rule is the identity (`min(b, window) == window`),
so sidecars written before the axis existed still compare equal — **no fingerprint or
on-disk format change**. The window floor previously made the sidecar inert: gemma4's
real window is 1024 and typical chat prompts are shorter, so nothing was ever backed
while capture still wrote K/V blocks no restore could read back.

Capture additionally refuses any non-bf16 cache (the snapshot type promises no
dtype). Text capture still refuses inherited media lineage. A native pure-image
capture is allowed only after global K/V capture succeeds, and only at a complete,
block-aligned checkpoint at or beyond the expanded image run; audio and mixed-media
turns remain disabled. Capture never anchors deeper than the K/V chain actually
reached (`PagedKVCacheAdapter::cold_captured_blocks`). It also only anchors where an
in-memory checkpoint already sits — and that chain reach lags the prompt badly,
because `capture_chain` stops at the first block the bounded writer queue refuses.
Measured on `Gemma-4-12B-IT-nvidia-mxfp-mlx` with an ~8.1k-token prompt under
`mlx agent`, it advanced ~34 blocks (544 tokens) per turn, reaching 1136 tokens by
turn 2 of an 8128-token prompt boundary — 508 blocks.

Image-bearing block identity is a versioned SHA-256 digest of the raw payload.
Every expanded placeholder contributes all four digest words plus its position to
the block's `extra_keys`, preceded by an image-key layout marker and a preprocessing
semantics marker. Later blocks inherit that identity through the parent hash chain.
Text-only blocks retain empty `extra_keys`, so their cold keys are unchanged. The
raw image must still be supplied after restart to reconstruct the digest, expansion
length and positions; once the exact global/sliding pair is restored beyond the
image span, the vision tower is skipped.

### The cold anchor rungs

The decode cadence alone cannot serve that: it fires once per `sliding_window`, so
its shallowest entry is a whole window, and a prompt several windows long ends the
prefill holding only its deepest couple of entries — the run above finished at
`{7168, 8128}` while the chain reached 1136, having created and then evicted the
entry at 1024. Nothing was ever written.

A persist turn therefore also publishes a fixed grid of **anchor rungs**,
`gemma4_sliding_cold_anchor_rungs`:

```text
block_size * 4^k          ->  {64, 256, 1024, 4096}   at block_size 16
```

pinned to **zero**, not to the prompt end (which is where qwen3_5's GDN ladder walks
from). A rung's cold key is the block chain over `tokens[0..b]`, so a grid anchored
at 0 makes the same sidecar object reusable by every later turn, and every later
process, whose prompt shares that prefix; a prompt-anchored ladder would land on
`112/496/2032` for one prompt and `128/512/2048` for the next and never dedup.

**Both** publishers use the grid: the prefill chunk walk
(`gemma4_sliding_chunk_checkpoint_boundaries`) and the decode step
(`gemma4_sliding_decode_boundary_plan`, whose predicate is the cadence UNION the
rungs). Decode has to, and not for symmetry's sake. The cadence is
`max(window, block).div_ceil(block) * block` = 1024 on the 12B, and `window / block_size`
is `64 = 4^3`, so every rung at `k >= 3` is also a cadence boundary and every rung
below the window is not. For the shape `mlx agent` actually sends — a short prompt and
a long generation — nothing else can ever publish `256`:

```text
turn 1 prefill 0..199   publishes {64}
turn 1 decode  200..N   cadence only: 1024, 2048, …        256 never fires
turn 2 prefill starts past 200      rung > start_offset  ->  256 refused
```

Capturing a rung is numerically transparent — `snapshot_from_attention_view` slices
the attention view the chunk already produced, and the chunk plan is **not** split at
a rung — so the whole cost is memory, which is why the grid is bounded two ways:

| bound                                       | value                                                                                                                                                                               |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GEMMA4_SLIDING_ANCHOR_MAX_RUNGS`           | 4 — how many rungs the grid may hold                                                                                                                                                |
| `GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES` | 3 GiB, at 4 bytes/element — the ceiling the rung grid is _admitted_ against, and the ceiling `trim_gemma4_sliding_prefix_checkpoints` _enforces_ over the entries actually retained |

Sizing is per boundary (`min(b, window)` rows), not per window; that is what makes the
two sub-window rungs nearly free and lets the fourth rung fit at all. On the 12B
geometry: rungs 41.9 + 167.8 + 671.1 + 671.1 MB, plus a 1342.2 MB reserve for the
pre-ladder entries, = 2894.1 MB of the 3221.2 MB budget (~1.4 GB actual bf16).

That admission arithmetic assumes the retained set is a PLANNED mix of cheap
sub-window rungs and a couple of deep entries, and nothing forces it to be. Once the
cursor is past one window every retained entry costs a full window, and six of those
are 4026.5 MB — 25% over the declared ceiling. So the `Ladder` arm enforces the budget
a second time, in bytes, after the count trim: it evicts until the summed
`min(b, window)` cost of the entries actually present fits. The count is a planning
figure; the byte loop is the guarantee. On unified memory that difference is not "a
cache tier degrades" — the extra gigabyte comes straight out of the weights and the
paged pool (see `docs/architecture.md`).

Retention answers to the same gate (`Gemma4SlidingRetentionPolicy`):

| turn                       | limit (12B)                      | victim                                                                                                                 |
| -------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| no cold tier — `PreLadder` | 2                                | oldest non-image-protected entry (unchanged); count only, never bytes                                                  |
| cold sidecar — `Ladder`    | 2 + 4 = 6, then the byte ceiling | oldest non-anchor; then the oldest anchor that is **not** an ancestor of the newest entry; then the **deepest** anchor |

The deepest-anchor step matters because the two image-protected prompt-boundary slots
are never eviction candidates, so a `{image, image, rung, rung, rung, deep}` store —
a VLM turn followed by a fresh text turn — leaves the first two steps with nothing to
take. Falling through to the pre-ladder floor there would evict the _shallowest_
entry, which is exactly the rung a chain advancing ~544 tokens per turn can reach.
Giving up the deepest anchor instead costs the least: the deep end is what the chain
reaches last.

The `PreLadder` arm is a compatibility contract, not an optimization: which checkpoint
a later warm turn lands on decides whether `prepare_gemma4_sliding_prefix` installs a
snapshot or replays the whole cached prefix through `run_sliding_only_prefill`, and
those are different spans of arithmetic that can emit different tokens. A
persistence-OFF request gets nothing back for that risk. The anchor-ancestor rule is
what stops a finished conversation's rungs from squatting after a lineage switch.

With the grid in place the 12B run above retains `{64, 256, 1024, 4096, 7168, 8128}`,
so the chain's reach finds an anchor from turn 1 and deepens
`256 -> 1024 -> 4096 -> …` as the writer queue drains. A capture that still finds no
usable anchor traces `sliding_cold_sidecar_capture_skipped` (now carrying `retained=`
and `anchor_rungs=`) rather than looking like a working cache.

For a text-only gemma4 prefix the sidecar is an **optimization, not a correctness
prerequisite**, and that is precisely what licenses scaling the boundary: a sliding
window is a _windowed_ state, so when the sidecar is absent
`run_sliding_only_prefill` reconstructs the missing rows from token ids exactly.
An image-crossing prefix is different: its placeholder ids cannot reconstruct the
vision embeddings, so restore accepts only a sidecar whose boundary exactly equals
the effective global K/V prefix; otherwise the prepared request restarts cold.
Contrast qwen3_5: a GDN recurrent state is a running summary of every preceding
token, valid ONLY at the exact boundary it was produced at, and recomputing it is
mathematically equivalent but not bit-identical (see below). That asymmetry is the
whole reason the scaled axis is gemma4's alone — `ColdSidecarPolicy::new` stays
unscaled and qwen3_5 / qwen3_5_moe are untouched.

### qwen3_5's GDN recurrent-state sidecar

`crates/mlx-core/src/models/qwen3_5/gdn_sidecar.rs` persists the two GDN tensors per
linear layer — `conv_state` `[1, K-1, conv_dim]` and `recurrent_state`
`[1, Hv, Dv, Dk]` — as **one concatenated blob per GDN layer** (`tensors_per_layer =
1`). The two tensors have different sizes and `ColdSidecarLayout` carries a single
`bytes_per_tensor`, so a two-tensor layout is impossible; the blob is
`conv ++ recurrent` and the layout's `dims` (`[K-1, conv_dim, Hv, Dv, Dk]`) give the
decoder the split offset. The element dtype is fixed at load from the pool cache
dtype and round-trips bit-exactly (`to_uint16_native` / `from_bfloat16` for 16-bit,
`to_float32` / `from_float32` for f32); capture re-checks each array's actual dtype
and skips rather than mislabel.

Unlike a sliding window, a GDN recurrent state is a running summary of **every**
preceding token (vLLM `MambaSpec`), so it is valid ONLY at the exact block-aligned
prefix length it was produced at — the `ColdSidecarPolicy` reconciles the restore
down to the deepest such boundary a validated sidecar backs, or to zero.

A prefill publishes a **ladder** of such boundaries, not one
(`gdn_prefill_checkpoint_boundaries`, `GDN_CHECKPOINT_LADDER_RUNGS = 4`,
`GDN_CHECKPOINT_LADDER_RATIO = 4`). The deepest rung is the `gdn_checkpoint_target`
(the largest full block strictly before the end of the prompt); each shallower rung is
a block-aligned quarter of the one above, and the ladder stops after four rungs or
when the next would fall to zero or below what is already cached. A 1400-token prompt
at `block_size` 16 therefore publishes `[16, 80, 336, 1392]`. The recurrent state is
only materialized at a rung when the prefill **splits** there, so when a GDN cold
policy is attached `paged_prefill` forces a split at **every** rung it crosses, even
under the default single-shot chunk size (`Qwen35Inner::cold_gdn_prefill_chunk_size`).

Why a ladder rather than the single endpoint boundary: capture may only anchor a
sidecar where the persisted K/V chain already reaches, and that chain advances by one
bounded writer queue's worth of blocks per turn. A single endpoint rung needs the
chain to reach the prompt's own end, which on a long prompt takes tens of turns; the
ladder needs it to reach only a quarter of the deepest rung. The ladder was designed
when that advance was pinned at 9 blocks per turn by `F_FULLFSYNC`; see
[Writer durability](#writer-durability-fsync2-not-f_fullfsync), which raised it but
did not remove the bound.

Be precise about what those splits cost. They are **mathematically equivalent, not
bit-identical**: every attention query still attends over the whole cumulative range,
but the GDN scan runs as one launch per rung crossed plus one, so the running state
takes an extra bf16 round trip at each boundary and the reduction order changes. The
chunk length is also the GEMM's `M`, so kernel selection can change with it. That is
the same tradeoff vLLM mandates for `mamba_cache_mode == "align"`, which hard-requires
chunked prefill (`model_executor/models/config.py`) and is the regime Qwen3-Next runs
in — so it is the reference design rather than an invention. It does mean the splits
are taken as soon as a policy is attached, i.e. on the FIRST persist-enabled run,
before anything has ever been restored.

The converse also has to hold, and it is enforced rather than assumed. A turn with
**no** GDN cold policy takes the break set it took before the ladder existed: the
single deep `gdn_checkpoint_target`, never the rungs. That is not cosmetic. Chunking
is on for a persist-off turn whenever `MLX_PAGED_PREFILL_CHUNK_SIZE` is positive, and
`packages/agent/src/run-agent.ts` plus `packages/cli/src/commands/launch-claude/index.ts`
both default it to 2048 unconditionally, before any persistence decision — so
`mlx agent --no-persist-cache` reaches a positive chunk size with no policy installed.
Without the guard a 1400-token prompt there would forward as 5 chunks
(`M` = 16, 64, 256, 1056, 8) where it used to forward as 2 (`M` = 1392, 8), which by
the paragraph above can change its sampled tokens. `paged_forward::prefill_checkpoint_boundaries`
takes the arm, `gdn_cold_sidecar_ladder_wanted` is the single predicate both it and
`cold_gdn_prefill_chunk_size` read, and
`gdn_checkpoint_tests::no_cold_policy_keeps_the_single_deep_boundary_the_ladder_replaced`
pins the values.

The restart-parity gate matched persist-on
against the persist-off baseline byte-for-byte on the gated checkpoints, which
**bounds** the divergence rather than proving it is absent at every prompt length —
and that remains exactly as true of the longer, up-to-four-split fixture the gate now
runs.

Capture is text-only in v1 and never anchors deeper than the K/V chain actually
reached (`cold_captured_blocks`). Within that reach it anchors at the **deepest ladder
rung**, so an early turn anchors shallow and later turns move the anchor down the
ladder as the chain's frontier advances. One sidecar per turn.

Only turn 1 publishes a full ladder. `gdn_prefill_checkpoint_boundaries` drops every
rung at or below `cached_prefix_len`, because this prefill never crosses those token
positions — `(4096, cached = 1008, block 16)` yields `[4080]` alone. So the shallow
rungs a later capture anchors on exist **only as residue from an earlier turn**, and
retention decides whether a warm conversation can still write a sidecar at all.

`gdn_checkpoint_store::prune_gdn_checkpoints` enforces two caps against that:
`GDN_PREFIX_CHECKPOINTS_PER_OWNER = 4` (the ladder width) per owner, and
`GDN_PREFIX_CHECKPOINT_LIMIT = 5` overall (one root session plus four concurrent pi
subagents). That second pairing crosses a language boundary — the fleet size is
`MAX_CONCURRENCY` in `packages/agent/src/extensions/subagent.ts` — so it is held by
a gate rather than by this sentence: `cold_tier::gdn_prefix_checkpoint_limit`
publishes the cap and `packages/agent/__test__/gdn-checkpoint-capacity.test.ts`
asserts `MAX_CONCURRENCY + 1` still fits under it.

Prune runs after **every** rung push, not once per ladder, so a store
already at the global cap sees a ladder arrive one rung at a time. Judged on
redundancy alone the newest rung's own predecessor is always the most redundant entry
present, so the global loop searches for a victim among every owner **except the one
publishing** first — otherwise the ladder eats itself down to the single endpoint rung
it exists to replace.

What a turn gets is decided by `gdn_retention_caps`, from the SAME predicate that
decides the break set (`gdn_cold_sidecar_ladder_wanted`) plus whether the root owner was
named. It returns the caps **and** the victim order together, because both move the same
observable quantity — the depth a later turn restores from — so neither may be taken
from a different column:

| `want_ladder` | explicit root | global cap | per-owner cap | victim order |
| ------------- | ------------- | ---------- | ------------- | ------------ |
| false         | false         | 2          | 2             | `PreLadder`  |
| false         | true          | 5          | 2             | `PreLadder`  |
| true          | false         | 4          | 4             | `Ladder`     |
| true          | true          | 5          | 4             | `Ladder`     |

The caps decide how many entries survive; the victim order decides **which**.
`Ladder` defers the publishing owner and then drops the rung with the smallest
next/own length ratio, which deliberately keeps a SHALLOW rung alive — those are the
only ones a cold restore can anchor on while the persisted K/V chain still lags the
prompt. `PreLadder` is 77e43031's order verbatim: the first same-owner ancestor in
publish order, no deferral. On a monotone conversation those keep different pairs
(`16, 32, 48` published → `PreLadder` keeps `32, 48`, `Ladder` keeps `16, 48`), so a
warm turn whose paged hit lands at 32 restores from a different depth under each.

`GDN_PREFIX_CHECKPOINT_LIMIT` is the global cap only when the root owner was named
explicitly, which today means `mlx agent`; every other caller leaves the root implicit,
and without a named root there is no separate global budget to spend, so the per-owner
cap is also the global one.

The `want_ladder = false` column is the same compatibility contract
`prefill_checkpoint_boundaries` keeps, for the same reason. With no cold GDN policy a
prefill publishes ONE rung, so the extra slots do not hold a publisher's own shallow
rungs — they hold whole **sibling lineages**, because several conversations multiplexed
over one model all publish under the same implicit owner id `""`. Whether the first
conversation's own entry survives until its next turn therefore depends on this number,
and the two outcomes build the recurrent state by different code:
`prepare_dense_gdn_prefix_state`'s `checkpoint` arm installs a snapshot the chunked
prefill took, its `replay_materialized` arm re-forwards the whole cached prefix through
`run_gdn_only_prefill_materialized`. Different span, different accumulation order,
different tokens. Measured on qwen3.5-0.8b-mlx-bf16, greedy, persistence **off**,
`MLX_PAGED_PREFILL_CHUNK_SIZE=2048` (what `mlx launch claude` sets unconditionally),
three conversations over one model, turn 2 of the first:

```text
  per-owner 4   state=checkpoint           restored 3584  replayed    0
  per-owner 2   state=replay_materialized  restored    0  replayed 3584
                -> emitted text diverges at character 56 of the same prompt
```

So persistence-off keeps the pre-ladder 2, and for the same reason keeps the pre-ladder
victim order — a cap restored without the order still retains a different SET, which the
next partial paged hit turns back into different tokens. Persist-on pays that drift
knowingly, in exchange for a restorable prefix; persistence-off would get nothing back
for it. The predicate may also flip mid-session when a cold tier is installed after some
turns have run, which is safe in both directions: 4 → 2 only prunes down at the next
publish, and 2 → 4 cannot resurrect an entry that is already gone.

Be exact about who survives that, because the guarantee is narrower than "the
publisher is protected". Preferring a foreign victim is a _preference_, not a floor:
the search only ever considers entries whose owner keeps something afterwards, so when
no other owner holds a spare rung — four siblings with one checkpoint each, exactly the
shape five slots was sized for — it finds nothing and the publisher's own ladder
collapses to its endpoint rung anyway, and that turn's cold capture misses.
`four_single_entry_siblings_outlive_the_publishers_ladder` pins that. The root, in
turn, keeps only its **last** checkpoint: the redundancy search carries no root guard,
so while the root holds more than one rung its ladder is the _preferred_ victim
(`one_subagent_turn_strips_the_root_to_its_deepest_rung`).

What the ordering cannot do is take an owner's warm reuse away. The only arm that
empties an owner runs after the redundancy search over _every_ owner comes back empty,
which means one entry per owner and still over cap — strictly more live owners than
slots. Below that point nobody goes blind under either order
(`one_subagent_turn_leaves_every_sibling_a_checkpoint`), and the measured hot-path cost
of the preference is zero: identical replayed prefix tokens and identical blind turns
in all 108 cells of `retention_sim::the_publisher_arm_never_costs_hot_path_replay`.
Past that point — six live owners in five slots — 28 of 40 agent turns re-forward their
whole cached prefix, under either order. That cliff is the count bound, not the victim
search, and moving it is a memory decision at ~75 MiB per slot.

`qwen3_5_moe` shares the identical GDN state type, sidecar module, and capture/replay
helpers, driven through `Qwen3_5MoeConfig::to_dense_config()`. That projection is safe
because it copies every GDN-relevant field verbatim (`linear_*` dims,
`full_attention_interval`, `num_layers`) and both configs define `is_linear_layer`
identically, so `gdn_layers()` resolves to the same layer set on either side;
MoE-ness affects only the MLP, which carries no cross-token state. Sharing a codec
with a passing family is not evidence, though, so the MoE ran its **own** restart-parity
gate on a real MoE checkpoint before joining the allowlist.

### Reading the counters from JS

Two native structs, two `#[napi]` readers, one JSONL line per turn:

```
ColdCacheStats    coldCacheStats()     hits misses enqueued queueDrops bytesWritten
  (the tier: reads                     bytesRestored evictions corruptions
   block-scoped, the                   writeErrors restoreDeclines
   write queue object-scoped)          + enabled root quotaBytes  (tier identity)

ColdSidecarStats  coldSidecarStats()   captureReached chainEmpty boundarySkips
  (out-of-pool                         alreadyPersisted enqueued queueDrops installed
   state)                              restoreSuppressed
        │
        └── per-turn delta ──> ~/.mlx-node/metrics/traces/<date>-<pid>.jsonl
                               cold<Field> / coldSidecar<Field>
```

`coldSidecarStats` is a **separate** reader on purpose. It never consults the tier, so
it reports on a run where the tier failed to open — which is exactly the run that needs
it. It is also why both prefixes exist.

The two `enqueued` / `queueDrops` pairs are **nested, not disjoint** — the one place the
scopes differ. `ColdCacheStats` answers _is the writer keeping up?_, so it counts every
object that took a slot in the shared queue, sidecars included; `ColdSidecarStats`
answers _did the family state persist?_, so it isolates the sidecars. **Summing them
double-counts every sidecar.** Read the block pair for queue health and the sidecar pair
for chain health. Scoping the queue counters to blocks would look tidier and would hide
the failure that matters most: a capture walk that fills the queue and starves the
sidecar leaves the whole turn's chain unrestorable, and only `ColdCacheStats.queueDrops`
reaches the dashboard. `ColdCacheStats.hits` and `bytesRestored` stay block-only, and
say so.

`installed` is the only read-side counter, and it is the one that cannot be inferred.
Every `install_*_cold_sidecar` early-return falls through to a full O(prefix) replay
that produces CORRECT state, so a regression from "restored and used" to "restored and
silently re-derived" leaves `text`, `num_tokens`, `cached_tokens`, `hits` and
`corruptions` all unchanged. `cold_tier_parity_harness.rs` asserts `restore_installs >= 1`
natively; `coldSidecarStats()` is the same fact for the dashboard and for `mlx agent`.

The capture counters split a zero-reuse run into its causes without `MLX_INFERENCE_TRACE`:

| line                                         | reading                                            |
| -------------------------------------------- | -------------------------------------------------- |
| `coldSidecarCaptureReached == 0`             | the turn's finalize never calls the capture        |
| `captureReached > 0`, `chainEmpty > 0`       | no whole block of the request was persisted        |
| `captureReached > 0`, `boundarySkips > 0`    | no retained checkpoint sat under the chain's reach |
| `captureReached > 0`, `alreadyPersisted > 0` | steady state — the chain is already on disk        |

That last row is why `alreadyPersisted` is a counter rather than silence: after the
first turn writes a rung, every later turn re-selects it and dedups, so `enqueued == 0`
is the signature of a HEALTHY repeated prompt as well as of a broken one.

`packages/agent/__test__/cold-counter-fields.test.ts` derives the JSONL field names from
both structs at runtime and pins them against `COLD_DELTA_FIELDS`, then drives a turn
through the provider and reads the file back. The field set narrowed twice before that
existed — four `ColdCacheStats` counters were dropped by the provider, and
`ColdSidecarTelemetry` had no binding at all — and neither narrowing breaks a type, a
build, or any test asserting on the fields that did survive.

### Restart-parity gate (authorizes an allowlist entry)

A family joins the allowlist only after `crates/mlx-core/tests/cold_tier_parity_harness.rs`
passes on a real checkpoint. The harness runs three instances — persist/capture,
fresh-instance restore, persist-off baseline — and asserts byte-identical `text`,
equal `num_tokens`, `hits > 0`, and `corruptions == 0` (so a fail-open restore cannot
masquerade as a pass).

Skipping is now only legal when no path was given. `MLX_TEST_MODEL_PATH` **unset**, or
set to an empty/whitespace value (a command substitution that produced nothing, a CI
`env:` fed by an empty expression), means the caller has no checkpoint and the gate
returns quietly. `MLX_TEST_MODEL_PATH` set to a non-empty path that is not a directory
**panics**: nobody sets a checkpoint path they do not mean, so that is a typo in an
invocation that believed it was gating the cold tier, and a green run there would
assert nothing at all.

The two qwen3_5 gates additionally set a `restore_prompt` — instances 2 and 3 run a
long prompt (1259 tokens on `qwen3.5-0.8b-mlx-bf16`, ladder `[16, 64, 304, 1248]`)
that shares a long body with the capture prompt and then diverges.
That makes the gate able to see the ladder at all: with one ~90-token prompt shared by
all three instances the ladder is `[16, 80]` and one turn's chain reach already covers
the deepest rung, so the restore always anchors at the prompt's end and a ladder
collapsed to a single endpoint boundary passes. With the divergent pair, the point
where the two prompts part caps `kv_chain_upper_bound` far below the capture prompt's
deepest rung, so the restore MUST reconcile onto a shallower one — which the harness
asserts (`cached_tokens` is a member of the capture prompt's ladder, and is strictly
below its deepest rung). This kills a one-rung ladder unconditionally; it does not pin
the rung count at four, and it does not pin the **ratio** either — 4 = 2², so halving
the spacing yields a ladder that still contains the rung a ~24-block chain would have
anchored on. Both gaps are closed by exact rung values in
`gdn_checkpoint_tests::ladder_rungs_are_quarters_of_the_one_above`, which is model-free
and runs in milliseconds. Arithmetic is pinned by arithmetic; the three-model-load gate
is reserved for the part only real weights can show — that a shallow rung is genuinely
written, found, decoded, and restored across a process boundary.

Note what the failure looks like when the ladder _does_ collapse: nothing is written at
all (the single endpoint rung is tens of blocks past what the writer queue drains), so
instance 2 restores zero and the harness's **assertion 1** fires with a message about
the restore path — not assertion 1b. The harness therefore prints the computed ladder
and the sidecar telemetry _before_ any assertion, and assertion 1's own message names
the prefill as the place to look.

```bash
MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/qwen3-0.6b-mlx-bf16 \
  cargo test -p mlx-core --test qwen3_cold_tier_parity -- --ignored --test-threads=1 --nocapture

MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/qwen3.5-0.8b-mlx-bf16 \
  cargo test -p mlx-core --test qwen3_5_cold_tier_parity -- --ignored --test-threads=1 --nocapture

MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/Qwen3.6-35B-A3B-mxfp4-mlx \
  cargo test -p mlx-core --test qwen3_5_moe_cold_tier_parity -- --ignored --test-threads=1 --nocapture

MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/lfm2.5-1.2b-thinking-mlx \
  cargo test -p mlx-core --test lfm2_cold_tier_parity -- --ignored --exact \
    --test-threads=1 --nocapture lfm2_cold_tier_restart_parity

MLX_PAGED_PREFILL_CHUNK_SIZE=64 MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/gemma-4-12b-it-qat-q4_0-mlx \
  cargo test -p mlx-core --test gemma4_grouped_cold_tier_parity -- --ignored --exact \
    --test-threads=1 --nocapture gemma4_grouped_cold_tier_restart_parity
```

The large-checkpoint gates are minutes, not hours. The latest local MoE run on
an M5 Max used the debug test profile with the checkpoint already resident in
the page cache:

| gate                                   | wall  | fresh model loads |
| -------------------------------------- | ----- | ----------------- |
| `qwen3_5_moe_cold_tier_restart_parity` | 721 s | 5                 |

This is a correctness gate, not a performance benchmark. Its wall time is dominated
by five loads of the 24.13 GiB MXFP4 weights and is sensitive to page-cache state and
build profile; do not use it to attribute a speedup to the cold-tier implementation.

The small-checkpoint gates run in CI, on the existing `model-test` matrix legs
that already download and convert the checkpoint they need (`.github/workflows/ci.yml`):
`qwen3_cold_tier_parity` on the `qwen3` leg, `qwen3_5_cold_tier_parity` plus the
Metal-gated unit tests `dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy`
and `moe_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy` on the
`qwen3_5-dense` leg, `lfm2_cold_tier_parity` on the LFM2.5 leg, and
`gemma4_grouped_cold_tier_parity` on the Gemma4 QAT PR leg. The MoE unit test rides
the dense Qwen leg because it needs no checkpoint at all — it builds a tiny synthetic
MoE and a real paged pool — and there is no MoE model leg for it to ride instead.

The two real-weights gates that stay local-only stay that way because of their
CHECKPOINTS, not their runtime:

- **MoE** — the smallest published `qwen3_5_moe` is 35B-A3B. The locally gated
  MXFP4 checkpoint contains 24.13 GiB of weight shards, past a standard macOS runner.
- **gemma4 MoE** — the standard CI runner covers dense E2B grouped restore. The
  local-only MoE gate uses a real 26B-A4B checkpoint; its grouped cache layout and
  codec are shared with dense, but the separate real gate protects the loader and
  KV-shared/MoE composition.

The structural fact both gemma4 gates rest on: with a `ColdSidecarPolicy` installed, a
non-zero `cached_tokens` in a _freshly loaded_ instance (empty hot cache) can only
have come from a validated sidecar. `gemma4_cold_tier_parity.rs` therefore runs **two**
gates, one per side of `min(boundary, window)`:

| gate                                         | prompt       | restored rotating state                         |
| -------------------------------------------- | ------------ | ----------------------------------------------- |
| `gemma4_cold_tier_restart_parity`            | ~1.2k tokens | post-wrap: a full window, `idx` inside the ring |
| `gemma4_cold_tier_restart_parity_sub_window` | ~350 tokens  | pre-wrap: `boundary` rows                       |

The long-prompt one keeps a `min_restored_tokens` floor of one whole window so it
stays in the `>= window` regime rather than quietly duplicating its sibling; that
floor is a statement about its fixture, not about what the layout can express. The
sub-window one adds the assertions `ChatResult` cannot carry — that the restored
sidecar backs exactly `cached_token_count` (so `aux_prefix_unbacked` never latches)
and that zero tokens were replayed — read off the `[MLX_TRACE]`
`sliding_prefix_prepare_done` / `paged_prefill_sliding_prefix_skipped` lines, which
is the only place that decision is observable from outside the crate. It reads that
channel rather than turning it on, so run it with `MLX_INFERENCE_TRACE=1` and
`MLX_INFERENCE_TRACE_FILE=...` supplied by the invocation — the module doc carries
the full command.

Witnessed on `Gemma-4-26B-A4B-IT-UD-Q3_K_XL-mlx`: `boundary=320` (well under the
1024 window), `replay_delta=0`, sliding prefill skipped as `already_primed`, restart
`cached=320` against `cached=0` on the persist-off baseline, `hits=120 misses=0
corruptions=0`, text and `num_tokens` identical across all three instances. The
~69 MB written against a ~200 MB full-window payload is the scaled axis doing its
job.

The tier manager is a process-global `OnceLock`, so every family gate is `#[ignore]`d
and must run with `--test-threads=1`.

## Relationship to `use_paged_attention`

`use_block_paged_cache` is independent of `use_paged_attention`. The latter drives the legacy `PagedKVCache` + `ContinuousBatchingScheduler` path used by the production server. Both can be on or off independently.

## Parity gate

Before `use_block_paged_cache` defaults can flip from `Some(false)` to enabled, every parity test below must pass on real weights:

| Test                                                            | Gate env var                     |
| --------------------------------------------------------------- | -------------------------------- |
| `crates/mlx-core/tests/qwen3_paged_vs_flat_parity.rs`           | `MLX_TEST_MODEL_PATH`            |
| `crates/mlx-core/tests/lfm2_paged_vs_flat_parity.rs`            | `MLX_TEST_MODEL_PATH`            |
| `crates/mlx-core/tests/gemma4_paged_vs_flat_parity.rs`          | `MLX_TEST_MODEL_PATH`            |
| `crates/mlx-core/tests/qwen3_5_paged_vs_flat_parity.rs`         | `MLX_TEST_MODEL_PATH`            |
| `crates/mlx-core/tests/qwen3_5_moe_paged_vs_flat_parity.rs`     | `MLX_TEST_MODEL_PATH`            |
| `crates/mlx-core/tests/nemotron_h_paged_vs_flat_parity.rs`      | `MLX_TEST_MODEL_PATH`            |
| `crates/mlx-core/tests/nemotron_h_concurrent_batched_parity.rs` | `MLX_TEST_NEMOTRON_H_MODEL_PATH` |
| `crates/mlx-core/tests/nemotron_h_mtp_midcycle_state.rs`        | `MLX_TEST_NEMOTRON_H_MODEL_PATH` |
| `__test__/models/qwen3-paged-parity.test.ts`                    | `QWEN3_PAGED_PARITY_MODEL_PATH`  |

The three NemotronH gates do **not** all read the same env var — `nemotron_h_paged_vs_flat_parity.rs`
(`resolve_source_model`) reads the generic `MLX_TEST_MODEL_PATH`, while
`nemotron_h_concurrent_batched_parity.rs` (`model_path`) and `nemotron_h_mtp_midcycle_state.rs`
(`resolve_source_model`) read the family-specific `MLX_TEST_NEMOTRON_H_MODEL_PATH`. Cited by
function name, not line: these files drift. Export both when running the family's full set — wiring
only one silently self-skips the others, and they early-return where libtest counts them as passed.

`nemotron_h_mtp_midcycle_state.rs` is the third gate, added alongside the MTP-owns-its-KV fix. It
covers the **mid-cycle-stop seam**.

_The scenario._ An MTP cycle commits up to `depth + 1` tokens, so at the pinned depth of 1 that is
up to two. The emit loop can stop _inside_ a cycle — a drafted-and-accepted EOS landing before the
cycle's last token. The feared failure was that the physical state (the flat KV trunk and all 23
Mamba-2 recurrent states) would end up advanced over the WHOLE cycle while only a prefix reached the
saved token history. Mamba-2 is non-invertible, so such a recurrent state could not be rewound in
place. **That failure does not occur at this family's pinned depth** — it was settled by measurement,
not by argument, and any older text asserting it as fact is retracted.

What actually happens is bounded. `verify_step` forwards the anchor plus the accepted drafts and
nothing else: the cycle's LAST outcome token — the bonus on a full accept, the residual on a
rejection — is sampled from a row that already exists and is never forwarded. A mid-cycle stop also
keeps every emitted token in the history (`mtp_history_drop_last` is `false` for a `"stop"` exit
whose last token is in the cache). So the trunk sits ahead of the history by exactly
`rollback_unemitted - 1`, which at depth 1 — where `rollback_unemitted` never exceeds 1 — is **zero**.
At depth 1 the bonus token is an `argmax` of a verify row that already exists; it is never pushed
back through the backbone, so a drafted-and-accepted EOS ends the turn with the flat KV offset EQUAL
to the saved history length. Measured on the v3 checkpoint:
`attention kv_offset - cached_token_history.len()` was **0 on every depth-1 probe** (history lengths
109 / 82 / 51 among them), including the turns that stopped mid-cycle with `unemitted == 1`; a
deliberately injected one-token skew moved the live-vs-cold Mamba-2 state distance by 35-150x, so
the zero is a sensitive probe rather than a blind one. The seam is real but the depth-1 skew is not;
the gate exists to keep it that way.

_The invariant._ The gate freezes observable outcomes, not an implementation. Two of them, both
mechanism-free:

1. **State matches history.** After every MTP turn the live attention KV offset must equal
   `cached_token_history.len()`. This is the seam's real contract and holds regardless of what
   guards exist.
2. **Warm equals cold.** After such a stop, a warm continuation on that session must be
   byte-identical to a fresh cold recompute of the identical transcript (`reset_caches()`, then
   decode the whole thing in one turn).

Any runtime behaviour that delivers those is acceptable — refusing prefix reuse onto an advanced
trunk, never letting the trunk run ahead in the first place, or an exact rewind. The machinery in
this area is the `flat_mtp_caches_desynced` latch (`NemotronHMtpStepper::rollback_unemitted` sets
it, `ChatBackend::flat_caches_desynced` reads it, both in
`crates/mlx-core/src/models/nemotron_h/model.rs`) and the generic flat flow's forced-`hit = 0`
branch (`crates/mlx-core/src/engine/session.rs`). The latch predicate is
`rollback_unemitted > 1`, so on a drafted-and-accepted depth-1 EOS **neither engages** — correctly,
because the trunk and the history are already aligned and latching would throw away the next flat
turn's whole prefix cache for nothing. The `> 1` arm is not dead: `mtp_adaptive_depth` routes cycle
depth through `AdaptiveDepthPolicy::pick_depth()`, which sweeps 1..=5 regardless of the
`depth: p.mtp_depth.min(1)` seed. Both assertions above hold either way, and the test file's module
docstring stays the authority. Symbols, not line numbers: these files are moving.

_The oracles._ Invariant 1 is the primary check because it is numeric. Around it sit a warm-reuse
liveness check (the following AR turn really does take the reuse arm, so invariant 2 is not
comparing two cold prefills), invariant 2 itself, an AR-twin check (at T=0 an MTP turn must
byte-match a pure-AR turn), and the flat-MTP -> paged-AR lane crossing. Read the file's module
docstring for the current, authoritative list — it also records that invariant 2 was MEASURED blind
at 1-token granularity on the v3 checkpoint, which is exactly why the numeric one leads. The gate
needs a checkpoint that actually carries an MTP head.

All Rust tests are `#[ignore]` and skip cleanly without the env var; the TS test uses `it.runIf`. Example invocation:

```bash
MLX_TEST_MODEL_PATH=./.cache/models/qwen3-0.6b-mlx-bf16 \
  cargo test -p mlx-core --test qwen3_paged_vs_flat_parity \
  -- --ignored --nocapture

QWEN3_PAGED_PARITY_MODEL_PATH=./.cache/models/qwen3-0.6b-mlx-bf16 \
  yarn vite run test __test__/models/qwen3-paged-parity.test.ts
```

Pass criteria: byte-equal `text` and `numTokens` between flat and paged on every prompt; byte-equal across a two-turn dialog (validates `find_cached_prefix` + `finalize_turn_keep_live` cross-turn semantics).

The TS test deliberately uses `chatSessionStart` rather than `generate` — `generate_sync` always uses fresh flat caches and never consults `paged_adapter`, so it would silently mask divergence.
