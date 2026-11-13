# Block-paged KV cache (vLLM-aligned)

A vLLM-style block-paged KV cache lives alongside the legacy flat `Vec<KVCache>` path. Multiple in-flight requests share refcounted KV blocks for any prompt prefix they have in common (system prompt, shared few-shot preamble, repeated tool-result frames, etc.).

Routing is per-model via the `use_block_paged_cache: Option<bool>` config field. Qwen3.5 dense and MoE ignore it: on Metal every turn runs on the pool, and a load off Metal has none (see below).
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

## Pool growth: the memory plan, runway, extents

The `qwen3_5` and `qwen3_5_moe` loaders size the paged pool from one memory
plan (`crates/mlx-core/src/engine/memory_plan.rs`, Splash
`evaluateEngineMemoryPlan`), evaluated after the weights are resident:

- **hard budget** = the Metal recommended working set less 15% and 1 GiB, or
  the `maxMemoryBytes` load option when that is lower;
- **fixed** = MLX active memory, other models' registered pools, restore
  staging and the pipeline/runtime reserves;
- the **pool max** is what the rest holds after one request's recurrent
  state and its peak transient, in whole blocks. The recurrent state lives
  outside the pool (Splash `availableForOneRequestKv = dynamic -
  laneStateBytes`); the scheduler still also charges it inside the pool, so
  the window pays it twice (54,425 on the M3 Pro inputs; 56,793 paid once).

The peak transient is one prefill chunk of at most **2,048 rows**
(Splash `ExecutionLimits::prefillTokenBudget`). Every Qwen3.5 dense and MoE
paged prefill, text and media, runs in chunks of that size
(`forward::paged_prefill_chunk_rows`); `MLX_PAGED_PREFILL_CHUNK_SIZE` can only
make the chunks smaller, a larger value is capped at 2,048. The GDN checkpoint
cuts come on top of the chunk cuts.

Each chunk, the first one and a cache hit alike, and each MTP or DFlash2
verify block stores its rows into the pool first and then attends in place
over the pool pages (`attention/paged_prefill_attention.metal`, Splash
`QwenTarget::addPrefillMixer`): 8-row tiles with the query group fused (two
parts at a group of 16), 32-token pages read through the block table, history
splits by Splash 1.6's rule and a reduce in split order
(`mlx_paged_attn::prefill_plan`), on int8 and bf16 pages alike. The chunk's
own rows are read back as stored, int8 · scale on an int8 pool. The plan
charges the kernel's workspace bound with the query staging, the output and
the chunk's K / V rows (Splash's `RuntimeArenas` attention tensors for one
lane), which does not grow with the window: 109,445,120 B for a 2,048-row 27B
chunk and 25,886,720 B for its 8-row verify. A pool needs a kernel instance (a
query group of 4, 6, 8 or 16, head 128 or 256, 16- or 32-token blocks); a
layout with no kernel fails the load ("no paged-attention kernel for
layout"). Every GPU family runs the kernel (Splash's Apple9 floor is
engine-wide, not this kernel's); M1 and M2 take the Apple9 pages-per-split
row, unmeasured there. A GPU whose compiler cannot build its pipelines (the
hosted macOS VM, "no known Apple GPU family"; asked once per process,
`device_prefill_mpp_builds`) attends the same stored rows through the plain
Metal varlen kernel instead (`gather_kv_for_prefill_chunk_varlen`). Splash has
no such fallback. That route is not bit-equal to the kernel (another
reduction order), and the plan does not charge its bf16 partials (about
`rows x Hq x ceil(ctx / 512) x (D + 2) x 4` B past 512 tokens of context; int8
tiles them to 2,048 row-partitions).

Every load starts the pool at the **runway**: 2048 tokens of blocks plus one
GDN cell (`qwen35_runway_blocks`, Splash `warmupKvPages`; 275 blocks on the
27B), and grows by whole extents of free blocks up to the max
(`PagedKVCacheAdapter::set_extent_growth`). Each grow is probed against the
plan's ceiling (hard budget less the reserves). A paged load whose working set
is unreadable is a load error, and so is a load whose footprint at the end
already exceeds the hard budget (Splash `auditActualMemory`). The load log
prints the plan (`memory plan:`), with `initial_blocks=`, `max_blocks=` and
`pool_window_tokens=`.

| Knob                    | Source                                                                        | Role                                                                                                                                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `paged_cache_memory_mb` | config.json                                                                   | MAX pool size in MiB, at most the plan's; above it the load fails and names the MiB the plan allows.                                                                                                           |
| `maxMemoryBytes`        | `Qwen35LoadOptions` (napi load), `--max-memory` on `mlx serve` / `mlx agent`  | Metal memory budget in bytes; it can only lower the automatic hard budget.                                                                                                                                     |
| `maxContextTokens`      | `Qwen35LoadOptions` (napi load), `--max-context` on `mlx serve` / `mlx agent` | Context token limit (1 to 1,010,000): caps every window; past the native window the target takes YaRN and the plan prices it; above the plan's window the load fails. The pool's size does not change with it. |

The pool always starts at the runway. The host overlays no Qwen3.5 config (`PagedConfigOverrideManager`
applies only to the `pagedModelTypes` it is given), so a source `paged_cache_memory_mb` reaches the plan's check. Other families size their pools on the same hard budget
(`automatic_hard_budget`) through `load_time_pool_sizing`.

**Residency.** A qwen3_5 / qwen3_5_moe / qwen4_exp pool keeps its K/V layer
buffers in a Metal residency set of its own (`LayerKVPool::hold_resident`,
`crates/mlx-paged-attn/src/residency.rs`; Splash `metal::Residency`). The set
is never attached to a command queue or a command buffer: a command makes
resident only what its encoders bind. Its members are the pool's live layer
buffers; a grow or a shrink adds the new pair and removes the old one in one
commit, so the set's bytes always equal the pool's. Adding or removing never
requests residency; one beat thread (`mlx-residency`) requests it every 500 ms
while the process-wide clock is within the keep-alive (`--idle-release`, default
10 minutes, since the last command), then ends it and dispatches one
`residency_kick` so Metal applies the end. A device without residency sets
leaves the pool unwired; other families' pools hold no set.

**Transient memory during grow.** `LayerKVPool::grow_to` grows one layer at
a time: it allocates the layer's larger pair, copies the old bytes, waits,
swaps the pair in and drops the old one before the next layer, so the peak is
the new pool plus one old layer (Splash grows by whole extents and copies
nothing; this is the closest layout without extents). The adapter drops its
cached MLX views before the grow so they do not keep old layers alive; the
headroom probe reserves one old layer (`pool_grow_reserved_bytes`). The
memory plan's pool max keeps one layer's bytes free for that copy. A grow that
fails part way keeps the grown layers (bytes intact, block count unchanged).
The growth notifier (`CacheLimitCoordinator::update_pool`) re-debits the
coordinator with the pool's measured buffer bytes, so the global cap tracks
the grown pool rather than the initial one.

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

On qwen3_5 / qwen3_5_moe the pool's K/V format follows `kvFormat` (Splash
`--kv-format`): `int8` pages hold int8 rows with one f32 scale per token, head
and side, `bf16` pages hold bf16 rows. Unset is int8 where the int8 kernels
serve the model (head 256, BF16 compute, 16- or 32-token blocks),
else bf16; an explicit `int8` the kernels cannot serve (head not 128 / 256,
another block size, f16 compute) fails the load. The cold tier stores an int8
block with its scales (`K | K scales | V | V scales` per layer, Splash slot
version 0), and the pool format is part of the cache fingerprint, so a bf16
cache is never read by an int8 load. qwen4_exp (Flash Next) follows
`kvFormat` the same way (unset: int8 at head 128 or 256, else bf16) on
Splash's Flash Next page (`QwenHybridLayout::kvLayout`, `SelectedBlocks`):
K and V token-major (`PageLayout::values_token_major`, the `_vt` store
instances) and a bf16 index plane of the indexer's pooled block keys in the
block (`PageLayout::index_rows`, Splash `PageStorage::indexOffset`), so its
block is the priced page (int8 423,936 B, bf16 811,008 B). Its own page
gathers read it (int8 dequantized to BF16); the paged-attention kernels, the
dense gather and the host transfers refuse such a page, and qwen4_exp has no
cold tier. Every other family's pool stays bf16.
There is no FP8 KV cache. `contextLimits().kvFormat` reports the pool's
format.

Every turn (autoregressive, vision, MTP, DFlash2) takes the pool lane's one
window. A load with MTP heads and no draft prices the MTP state it holds at the
window into that lane (Splash `laneStateBytes`), beside the pool, and inside
the pool as well when the continuous-batching scheduler books it there. A load
with a DFlash2 draft prices the draft state (a second GDN cell and the draft's
K/V ring, Splash `CompositeStateLayout::laneBytes`) beside the pool, never in
it: a DFlash2 turn runs alone on the Barrier lane. Its prefill and its fixed
8-row verify read and write the pool; a load with a draft and no pool fails.
`mlx eval` reads the pool on a paged load, the same route a served turn reads.

| Model             | Default | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------- | :-----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Qwen3**         | **on**  | Greedy + prefix-reuse byte-equal vs. flat path on Qwen3-0.6B BF16. Opt out via `use_block_paged_cache: Some(false)`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **LFM2.5**        | **on**  | Same parity result on LFM2.5-1.2B. Hybrid arch — only `full_attention` layers go through the adapter; conv layers stay on `Lfm2LayerCache::Conv`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Gemma4**        | **on**  | Serial/concurrent start and continuation parity on Gemma-4-E2B-IT with a real occupancy-2 wave. Full and sliding layers use distinct paged groups; expired sliding blocks are replaced by a null sentinel; KV-shared layers alias their global/sliding anchor. Same-owner continuation is live; cross-owner prefix hits currently fail closed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Qwen3.5 Dense** | **on**  | Single-turn greedy parity and paged construction are verified on Qwen3.5-0.8B BF16. Full-attention K/V is paged; GDN recurrent state remains request-local and is checkpointed beside K/V for SSD restore. On Metal every turn runs on the pool: there is no flat lane and no switch, on every Apple GPU family; a layout with no paged-attention kernel fails the load with the reason. Off Metal (the CUDA preview) the load has no pool and serves bf16 autoregressive turns on the flat cache.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Qwen3.5 MoE**   | **on**  | Uses the same paged full-attention, DFlash2 SSD states, and two-row text scheduler contract as dense. Sparse expert routing stays batched over `[N,1,H]`; checkpoint-specific projections preserve exact greedy parity while K/V gather/attention remains batched. MTP/media turns remain exclusive. Real `Qwen3.6-35B-A3B-mxfp4-mlx` decode gates remain local-only because no small published checkpoint fits the standard CI runner; synthetic Metal gates cover paged construction and N=2 token parity in CI.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **NemotronH**     | **on**  | Hybrid Mamba-2 SSM + GQA + MoE-FFN: only the six `full_attention` layers route through the adapter (32 Q / 2 KV heads, head_dim 128 — the generic batched paged route, no per-family kernel). Mamba recurrent state stays request-local (per-request `[3,6144]` conv + `[64,64,128]` SSM f32 rows per SSM layer, stacked/scattered around each batched decode) and is checkpointed only in-process (no SSD sidecar yet — the family is NOT cold-restore eligible). The scheduled lane is default-on, but the fused `[N,1]` decode is reachable only on a dense/bf16 checkpoint: `load_inner` (`persistence.rs`) sets `row_exact_decode_projections` for any quantized checkpoint — including the only published one — so the decode wave dispatches to `run_row_exact_decode_wave` (`model.rs`), N serial single-row decodes (bit-identical to serial, but N full weight streams), and the fused branch is dead code there. `run_scheduled_prefill_slice` re-splits each scheduled range on `config.chunk_size` (128 on the released checkpoint), so boundaries _inside_ one scheduled slice are chunk multiples; the slice's own start/end come from the engine's plain 2048-token grid minus that step's decode rows and can land mid-chunk, which the chunk scan absorbs by padding the final chunk (reduction order changes, semantics do not). A **synchronous** MTP turn stays on the exclusive ordered lane and runs the FLAT speculative core: the family declares `supports_paged_attention: false`, and `TurnPlan::resolve` gives a flat-only decoder the flat lane WITH its target rather than dropping it to paged AR. A **streaming** MTP turn has no flat arm to reach (`supports_streaming: false`, `model.rs`), so it plans as plain paged AR and keeps its scheduled slot. Symbols, not line numbers, throughout this row: `models/nemotron_h/` is under active change. All **three** real-checkpoint gates are listed on the `nemotron_h` leg of the `model-e2e` matrix (see the parity-gate table below), but that leg has never executed: the `model-test` job is gated on the `model-e2e` PR label and this leg's run step additionally exits early on push/dispatch, and its bill — ~38 GB of disk for source + converted output, plus ~42 GB resident for the two model copies `nemotron_h_paged_vs_flat_parity` holds at once — exceeds a hosted runner, so local runs remain the only evidence; synthetic Metal gates cover paged construction, N=2 batched==serial T=0 parity, and the break-set boundary rule in CI on every run. |
| **Muse-Glimmer**  | **on**  | Grouped full/sliding paged KV through the gemma4-shaped coordinator (`Gemma4KVCacheCoordinator`); opt out via `use_block_paged_cache: Some(false)`. DFlash draft caches are off-pool flat and are never persisted. The SSD sidecar uses the `SlidingWindow` codec shared with gemma4, at one joint full+sliding boundary, fail-closed. Preempt-to-SSD is disabled — `scheduler_has_cold_tier` returns false, so a preempted victim always recomputes — while cold-restore is allowlisted (`COLD_RESTORE_FAMILIES` includes `muse_glimmer`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

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

Fixed-depth text DSpark verifies through the resident paged groups and keeps
its draft context per request. Assistant and other flat-only draft routes use a
request-local flat owner lane while their command executes; loading a draft does
not turn the paged coordinator off. Ordinary owners use the resident grouped pools. A live owner may
not switch cache layouts mid-history, and releasing the owner drops both lane
registries plus metadata.

For Qwen3.5 (dense + MoE) the paged decode path (and the off-Metal flat one)
is pure-Rust eager — the compiled C++ forward and its process-wide locks
(`DENSE_COMPILED_MUTEX` / `COMPILED_WEIGHTS_RWLOCK`) were deleted in the
chat-engine refactor (`ee88b92b`), so there is no compile state to corrupt and
no per-step lock is taken on either path. VLM checkpoints run their image
turns through the paged vision core when `paged_adapter` is present (plain AR,
MTP weights ignored); a load with no pool publishes no image capability, so
the engine refuses an image turn before it renders.

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

vLLM's offload stack, for contrast, is explicit cross-address-space tiers — CPU pinned-mmap primary;
fs/obj/p2p secondary (`vllm/v1/kv_offload`). Unified memory collapses the GPU↔CPU hop, so this SSD
tier corresponds to their **fs** secondary tier, with the durability/integrity/quota/fingerprint
contrasts in [architecture.md](architecture.md). Their analog of the sidecar reconcile-down below is
the eagle-group volatile-trailing-chunk exclusion, applied on both store and load
(`kv_connector/v1/offloading/scheduler.py`). Their v1 preemption is recompute-by-contract — free the
blocks, zero the counter — but freed block hashes linger in the device prefix cache until reuse, and
an offloading connector turns resume into a partial load: vLLM recomputes and lets caches shortcut,
where mlx-node makes a measured spill-vs-recompute decision at preemption (`select_preemption_mode`).

Dense `qwen3` is sound because its pool covers **all** layers, so a restored block
reconstructs the whole prefix. Every supported hybrid family now pairs K/V with an
exact-boundary sidecar or a second paged group:

| Family              | State outside the primary full-attention pool              | Restore companion                                                                   | Restore-eligible |
| ------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------- | :--------------: |
| `qwen3` (dense)     | none — pool covers every layer                             | n/a                                                                                 |     **yes**      |
| `gemma4`            | sliding-attention paged groups                             | grouped live-window K/V sidecar; all groups install at one boundary                 |     **yes**      |
| `muse_glimmer`      | sliding-attention paged groups (gemma4-shaped coordinator) | grouped `SlidingWindow` sidecar sharing gemma4's codec; joint full+sliding boundary |     **yes**      |
| `qwen3_5` (dense)   | GDN recurrent state                                        | none: DFlash2 states on SSD instead (below)                                         |      **no**      |
| `qwen3_5_moe`       | GDN recurrent state (same as dense)                        | none: DFlash2 states on SSD instead (below)                                         |      **no**      |
| `lfm2` / `lfm2_moe` | ShortConv recurrent state                                  | exact-boundary `ConvState` sidecar                                                  |     **yes**      |
| `nemotron_h`        | Mamba-2 SSM recurrent state                                | none yet (in-process only)                                                          |      **no**      |

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

A one-token decode read of a sliding group (Gemma4, Muse-Glimmer) starts at the
group's first live block and passes the context length from there
(`decode_read_span` in `paged_kv_cache_adapter.rs`); the kernel masks the rest of
the window from the partial first page. The graph-native and the synchronous
decode routes share that span, so the paged kernel cuts the same keys into the
same partitions on either route and returns the same bits.

The older rotating-cache representation remains active for the flat/exclusive
route used by the flat owner lanes. It shares the same sidecar layout
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

### DFlash2 states on SSD (qwen3_5 / qwen3_5_moe)

Qwen3.5 dense and MoE are not on the K/V restore allowlist. Like Splash, their SSD
tier holds **DFlash2 states**, not per-turn K/V: a turn writes nothing to disk, and a
K/V chain alone never backs a prefix (`ColdContextBacking::StateOnly`). The tier opens
for every paged load, with a DFlash2 draft or without one. On a load without a draft,
text turns publish the same restore points as Splash (`publishReachedStateBoundaries`
runs for every request): a checkpoint every 4,096 prompt tokens (twice the 2,048-token
state window) and the replay point at the last whole block before the generation
prompt. That holds for autoregressive and native-MTP turns alike, whole-turn or
scheduled: the MTP prefill reaches the points in the same forward that seeds its
drafter, and an MTP turn resumes from a RAM or disk state as a plain one does. Their
objects hold the GDN cell alone. On a load with a draft, a whole autoregressive turn
(speculation off for the request) primes and prefills as a DFlash2 turn does: its
states carry the draft window, its lane is released at its end, and a suspended one
resumes from its deepest state. Image turns still keep their states in the
in-process GDN ladder below, not on the tier: the state label holds tokens and salt
but no image keys, so a restart could not tell two images apart.
`tests/qwen3_5_state_tier_restart.rs`
gates a restart on the default `mlx serve` path (a streamed MTP turn) and on a plain
autoregressive one.

| What                 | Splash                                                                                                                            | mlx-node                                                                                                                                                                                                                                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Object               | `SlotFile` slot of `stateSpans`                                                                                                   | one `ColdGroup::DFlash2State` object: the GDN cell, then (with a draft) its 4-bit context window; its label carries the lineage tokens, salt, stamp and class                                                                                                                                                                    |
| Quota                | `--max-cache-disk auto`: min(32 GiB, a quarter of the room beyond the floor)                                                      | the same (`disk_tier`); a state is charged its payload (K8)                                                                                                                                                                                                                                                                      |
| Default              | temporary, gone with the process                                                                                                  | persistent (`<cache-dir>/<namespace>`) for `mlx serve`, `mlx agent`, `mlx launch claude` and the desktop app; a locked `$TMPDIR/mlx-node-cache-*` directory under `--no-persistent-cache` and for a library `loadModel` without `persistentCache`: `closeCache` removes it at a clean stop, the next start removes a crashed one |
| `--persistent-cache` | per-namespace directory + lock, adopted at the next start                                                                         | `<cache-dir>/<namespace>` (default root `~/.mlx-node/cache/prefix-cache`); the namespace hashes the target, the draft (or none) and the cache geometry                                                                                                                                                                           |
| Writes               | reclaim of a RAM state (`StateCache::reclaim`), write-behind (`WriteBehind`), a capture the ledger refused (`publishStateToDisk`) | the same three; one staging buffer, one write in flight; a state's K/V chain follows it to disk before the state counts as landed                                                                                                                                                                                                |
| Restore              | `acquireDeepest` over RAM and disk, read into the lane, promotion is Optional and Immediate                                       | the same; a failed read discards the state and the turn resumes from the next one down                                                                                                                                                                                                                                           |
| Restart              | `Cache::adopt`: a state comes back only with its whole K/V chain                                                                  | `adopt_all` at load                                                                                                                                                                                                                                                                                                              |
| Clean stop           | `flushRestorePoints` within `kFlush` (6 s)                                                                                        | the host's `close()` gives every loaded model's `closeCache` one shared 6 s deadline (`FLUSH_MS`), then drains the process tier of the other families                                                                                                                                                                            |

Flags: `mlx serve --max-cache-disk <auto|SIZE|0> --[no-]persistent-cache --cache-dir <dir>`
(`mlx agent` forwards the same flags). The cache is persistent unless `--no-persistent-cache`
or `--max-cache-disk 0` is given; Splash defaults to a temporary one. `/health` reports the tier under
`memory_governor.dflash2_states.disk` (`mode` `temp` / `persistent` / `off`).

The in-process GDN checkpoint store (`gdn_checkpoint_store`) is unchanged by the SSD
tier: it keeps two entries per owner, five in all when the root owner is named.

### Reading the counters from JS

Two native structs, two `#[napi]` readers, one JSONL line per turn:

```
ColdCacheStats    coldCacheStats()     hits misses enqueued queueDrops bytesWritten
  (the tier: reads                     bytesRestored evictions corruptions
   block-scoped, the                   writeErrors restoreDeclines
   write queue object-scoped)          + enabled root quotaBytes  (tier identity)

ColdSidecarStats  coldSidecarStats()   captureReached chainEmpty boundarySkips
  (out-of-pool                         alreadyPersisted enqueued queueDrops installed
   state)
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

```bash
MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/qwen3-0.6b-mlx-bf16 \
  cargo test -p mlx-core --test qwen3_cold_tier_parity -- --ignored --test-threads=1 --nocapture

MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/lfm2.5-1.2b-thinking-mlx \
  cargo test -p mlx-core --test lfm2_cold_tier_parity -- --ignored --exact \
    --test-threads=1 --nocapture lfm2_cold_tier_restart_parity

MLX_PAGED_PREFILL_CHUNK_SIZE=64 MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/gemma-4-12b-it-qat-q4_0-mlx \
  cargo test -p mlx-core --test gemma4_grouped_cold_tier_parity -- --ignored --exact \
    --test-threads=1 --nocapture gemma4_grouped_cold_tier_restart_parity
```

The small-checkpoint gates run in CI, on the existing `model-test` matrix legs
that already download and convert the checkpoint they need (`.github/workflows/ci.yml`):
`qwen3_cold_tier_parity` on the `qwen3` leg, `lfm2_cold_tier_parity` on the LFM2.5
leg, and `gemma4_grouped_cold_tier_parity` on the Gemma4 QAT PR leg.

The real-weights gate that stays local-only stays that way because of its
CHECKPOINT, not its runtime:

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
