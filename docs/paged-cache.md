# Block-paged KV cache (vLLM-aligned)

A vLLM-style block-paged KV cache lives alongside the legacy flat `Vec<KVCache>` path. Multiple in-flight requests share refcounted KV blocks for any prompt prefix they have in common (system prompt, shared few-shot preamble, repeated tool-result frames, etc.).

Routing is per-model via the `use_block_paged_cache: Option<bool>` config field. Only the full-attention layers of supported models go through the paged adapter — sliding-window, convolutional, and recurrent (GDN) layers stay on their dedicated cache types regardless of the flag.

## Foundation types

| Type                  | Location                                                    | Role                                                                                                                                                                                                |
| --------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BlockAllocator`      | `crates/mlx-paged-attn/src/block_allocator.rs`              | Logical lifecycle — per-block refcounts, LRU eviction, prefix-hash table for cross-request reuse                                                                                                    |
| `LayerKVPool`         | `crates/mlx-paged-attn/src/layer_kv_pool.rs`                | Physical storage — per-layer Metal K and V `Buffer` pairs sized to `paged_cache_memory_mb`                                                                                                          |
| `PagedKVCacheAdapter` | `crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs` | Session-friendly wrapper. Per-request lifecycle: `reset_for_new_request` → `find_cached_prefix` → `allocate_suffix_blocks` → `record_tokens` → `register_full_blocks_for_reuse` → `release_request` |

`BlockAllocator` and `LayerKVPool` are intentionally split so the legacy `CacheEngineManager` path (used by `use_paged_attention`, a different flag — see below) is unaffected. `paged_cache_memory_mb` defaults to 2048 when `None`.

## Per-model support matrix

| Model             | Default | Status                                                                                                                                                                                                                                          |
| ----------------- | :-----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qwen3**         | **on**  | Greedy + prefix-reuse byte-equal vs. flat path on Qwen3-0.6B BF16. Opt out via `use_block_paged_cache: Some(false)`.                                                                                                                            |
| **LFM2.5**        | **on**  | Same parity result on LFM2.5-1.2B. Hybrid arch — only `full_attention` layers go through the adapter; conv layers stay on `Lfm2LayerCache::Conv`.                                                                                               |
| **Gemma4**        | **on**  | Same parity result on Gemma-4-E2B-IT. Sliding layers stay on `RotatingKVCache`; global layers go through the adapter; KV-shared layers consume the anchor via `SharedOnGlobal` / `SharedOnSliding`.                                             |
| **Qwen3.5 Dense** | **off** | Single-turn greedy parity verified on Qwen3.5-0.8B BF16. Default-flip pending a perf decision against the compiled C++ flat path. GDN linear-attention layers stay on flat `ArraysCache` (no cross-request reuse — vLLM `MambaManager` stance). |
| **Qwen3.5 MoE**   | **off** | Forward dispatch wired and parity-test scaffold present, but no local MoE checkpoint to verify against yet.                                                                                                                                     |

For Qwen3.5 (dense + MoE) the per-dispatch-site **compile lockout** is critical: every chat-entry site (`chat_sync_core`, `chat_tokens_delta_sync`, `chat_stream_sync_inner`, `chat_stream_tokens_delta_sync_inner`) early-returns into the paged variant **before** acquiring `DENSE_COMPILED_MUTEX` / `COMPILED_WEIGHTS_RWLOCK`, so flat-path turns and paged-path turns can interleave without corrupting compiled state. VLM checkpoints are permitted under a text-only contract — image-bearing turns fail loudly when `paged_adapter.is_some()`.

## SSD cold tier: hybrid families and the auxiliary sidecar

The cold tier (`crates/mlx-paged-attn/src/cold_cache.rs`) persists full paged K/V
blocks so a warm prefix survives a process restart. Whether a family may _restore_
from it is a correctness decision, gated by an allowlist that exists in two places
and is drift-guarded by a test:

| Side  | Symbol                                                                      |
| ----- | --------------------------------------------------------------------------- |
| Rust  | `COLD_RESTORE_FAMILIES` in `crates/mlx-core/src/cold_tier.rs`               |
| TS    | `COLD_TIER_RESTORE_FAMILIES` in `packages/agent/src/provider/model-host.ts` |
| Guard | `packages/agent/__test__/cold-tier-families.test.ts`                        |

Dense `qwen3` is sound because its pool covers **all** layers, so a restored block
reconstructs the whole prefix. Every other supported family is **hybrid** — it sizes
the pool to attention layers only and keeps the rest of its cross-token state
outside:

| Family              | Out-of-pool state                        | Sidecar | Restore-eligible |
| ------------------- | ---------------------------------------- | ------- | ---------------- |
| `qwen3` (dense)     | none — pool covers every layer           | n/a     | **yes**          |
| `gemma4`            | sliding-window `RotatingKVCache`         | yes     | **yes**          |
| `qwen3_5` (dense)   | GDN (gated delta-net) recurrent state    | yes     | **yes**          |
| `qwen3_5_moe`       | GDN recurrent state (same as dense)      | yes     | **yes**          |
| `lfm2` / `lfm2_moe` | short-conv state (no serialization path) | no      | no               |

The allowlist is enforced **natively**, not only in the agent:
`cold_tier::resolve_persist_cold` consults `cold_restore_supported(model_type)`
before any other signal, so a family that is off the list never persists or restores
— not under an explicit `persist_paged_cache` config, not under
`MLX_PERSIST_PAGED_CACHE=1`, not via a direct library caller that bypasses the agent.
A loader may therefore carry a fully wired cold bracket ahead of proving it; the gate
keeps that bracket dormant until the family is admitted.

A K/V-only restore for a hybrid would resume from state the pool never held. Two
mechanisms make that impossible rather than merely unlikely.

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
| ------------------ | ----------------- | ----------------- | --------------- | --------------- |
| 0.10 ms            | 9                 | 11, 13            | 9               | 11, 10          |
| **0.20 ms (real)** | 9                 | **28, 25**        | 9               | **17, 18**      |
| 0.32 ms            | 9                 | 600+, 403         | 9               | 72, 45          |

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

|                    | process kill | kernel panic | sudden power loss / hard reset |
| ------------------ | ------------ | ------------ | ------------------------------ |
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
dense block `payload_checksum` is 0.061 ms of a 0.432 ms `Tw`, so making it *free*
(the harness prints that projection) would reach only 0.366 ms — worth ~3 more blocks
per turn, against invalidating every cache on disk.

`Q` — `DEFAULT_QUEUE_DEPTH`, 8 — is now the binding term again, and it is untouched
here. `N` cannot run far past `Q + 1` until `Tw` falls under `Tc`, and at the measured
`Tc` (~0.21 ms) it has not: `Tw` is ~0.31-0.43 ms. Raising `Q` is the next lever, and
cutting `Tw` ~10x is what makes it affordable — the backlog a larger queue admits now
drains an order of magnitude faster, and `drain()` at agent exit is bounded by the same
`Tw`.

**What happens if `Tw` does fall under `Tc`**, on a `MLX_COLD_CACHE_DIR` pointed at a
RAM disk or a drive faster than the one measured: the queue stops refusing, `N` has no
frontier, and `capture_chain` walks **every** full block of the request in one turn.
That is the intent of the walk — the whole prompt persists on its first turn — but the
cost lands on the inference thread as one blocking `read_block_all_layers` per new
block, after the turn's last token. It is bounded by prompt length, not by anything
tunable: a first turn on a 64 K-token prompt is 4096 blocks x ~0.2 ms = **~0.9 s of
added turn tail**, and nothing counts or caps it. The margin against that today is
`Tw/Tc` ~ 2x on the measured pair, and the `fsync(2)` `Tw` above is an over-estimate,
so the real margin is smaller than 2x. Whoever raises `Q` should put a cap and a
counter on the walk in the same change.

### gemma4's sliding-window sidecar

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

Capture additionally refuses any non-bf16 cache (the snapshot type promises no dtype)
and any media turn (v1), and never anchors deeper than the K/V chain actually reached
(`PagedKVCacheAdapter::cold_captured_blocks`). It also only anchors where an
in-memory checkpoint already sits, and below one window the sole such anchor is the
prompt-boundary checkpoint (above a window the decode cadence supplies its own).
Where that one *lands* decides whether anything can use it. A prefill over `N`
tokens stores exactly one, and restore asks for at most `N - 1` tokens:

| prompt length          | checkpoint lands at            | usable |
| ---------------------- | ------------------------------ | ------ |
| `N % block_size != 0`  | `floor(N / block_size) * bs`   | yes    |
| `N % block_size == 0`  | `N` — one block out of reach   | no     |

So with `block_size = 16` a 500-token prompt anchors at 496 while a 512-token one
anchors at 512 and gets nothing. A capture that finds no usable anchor traces
`sliding_cold_sidecar_capture_skipped` rather than looking like a working cache.

For gemma4 the sidecar is an **optimization, not a correctness prerequisite**, and
that is precisely what licenses scaling the boundary: a sliding window is a
_windowed_ state, so when the sidecar is absent — or is only representable shallower
than the K/V prefix — `run_sliding_only_prefill` reconstructs the missing rows from
token ids exactly, and the sidecar only buys back that replay. Contrast qwen3_5:
a GDN recurrent state is a running summary of every preceding token, valid ONLY at
the exact boundary it was produced at, and recomputing it is mathematically
equivalent but not bit-identical (see below). That asymmetry is the whole reason the
scaled axis is gemma4's alone — `ColdSidecarPolicy::new` stays unscaled and qwen3_5 /
qwen3_5_moe are untouched.

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
| --- | --- | --- | --- | --- |
| false | false | 2 | 2 | `PreLadder` |
| false | true  | 5 | 2 | `PreLadder` |
| true  | false | 4 | 4 | `Ladder` |
| true  | true  | 5 | 4 | `Ladder` |

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
publisher is protected". Preferring a foreign victim is a *preference*, not a floor:
the search only ever considers entries whose owner keeps something afterwards, so when
no other owner holds a spare rung — four siblings with one checkpoint each, exactly the
shape five slots was sized for — it finds nothing and the publisher's own ladder
collapses to its endpoint rung anyway, and that turn's cold capture misses.
`four_single_entry_siblings_outlive_the_publishers_ladder` pins that. The root, in
turn, keeps only its **last** checkpoint: the redundancy search carries no root guard,
so while the root holds more than one rung its ladder is the *preferred* victim
(`one_subagent_turn_strips_the_root_to_its_deepest_rung`).

What the ordering cannot do is take an owner's warm reuse away. The only arm that
empties an owner runs after the redundancy search over *every* owner comes back empty,
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

Note what the failure looks like when the ladder *does* collapse: nothing is written at
all (the single endpoint rung is tens of blocks past what the writer queue drains), so
instance 2 restores zero and the harness's **assertion 1** fires with a message about
the restore path — not assertion 1b. The harness therefore prints the computed ladder
and the sidecar telemetry *before* any assertion, and assertion 1's own message names
the prefill as the place to look.

```bash
MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/qwen3-0.6b-mlx-bf16 \
  cargo test -p mlx-core --test qwen3_cold_tier_parity -- --ignored --test-threads=1 --nocapture

MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/Gemma-4-26B-A4B-IT-UD-Q3_K_XL-mlx \
  cargo test -p mlx-core --test gemma4_cold_tier_parity -- --ignored --test-threads=1 --nocapture

MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/qwen3.5-0.8b-mlx-bf16 \
  cargo test -p mlx-core --test qwen3_5_cold_tier_parity -- --ignored --test-threads=1 --nocapture

MLX_COLD_CACHE_DIR=$(mktemp -d) \
  MLX_TEST_MODEL_PATH=~/.mlx-node/models/Qwen3.6-35b-a3b-UD-Q2_K_XL-mlx \
  cargo test -p mlx-core --test qwen3_5_moe_cold_tier_parity -- --ignored --test-threads=1 --nocapture
```

The large-checkpoint gates are minutes, not hours. Measured on an M5 Max, `--release`,
with the checkpoint already resident in the page cache:

| gate                                          | wall     | fresh model loads |
| --------------------------------------------- | -------- | ----------------- |
| `qwen3_5_moe_cold_tier_restart_parity`         | 105 s    | 5                 |
| `gemma4_cold_tier_restart_parity`              | 175 s    | 15                |
| `gemma4_cold_tier_restart_parity_sub_window`   | 84 s     | 9                 |

Earlier notes here said ~66 min and ~26 min. **Do not read the drop as something the
cold-tier commits bought** — the arithmetic does not support it. The long gemma4 gate
restores 739 blocks; at the batched command buffer's own measured saving that is
`739 x (5.271 - 1.158) ms` = **3.0 s**, and the capture side is smaller again. A few
seconds, against a 62-minute difference.

What the two numbers really differ in is model-load I/O. Neither checkpoint carries a
`.mlx-download-complete.json` marker, so every persist-enabled load full-shard-hashes
~14 GB of weights, and the long gate makes 15 such loads. 175 s over 15 loads is
11.7 s per load, with the weights already in the page cache. 66 min over the same 15
loads is 264 s per load — about what ~14 GB reads like when it has to come off the
disk each time. Both runs keep their checkpoints on the same volume, so page-cache
state is the variable, not the storage. So: **budget the table above only for a warm
page cache, and expect minutes per load otherwise.** The 264 s/load figure is what the
arithmetic implies, not a re-measured number — the cold case was not re-run. What is
certain is the direction: the cold-tier commits moved seconds, not the headline.

The two small-checkpoint gates run in CI, on the existing `model-test` matrix legs
that already download and convert the checkpoint they need (`.github/workflows/ci.yml`):
`qwen3_cold_tier_parity` on the `qwen3` leg, `qwen3_5_cold_tier_parity` plus the
Metal-gated unit tests `dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy`
and `moe_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy` on the
`qwen3_5-dense` leg. The MoE unit test rides that leg because it needs no checkpoint
at all — it builds a tiny synthetic MoE and a real paged pool — and there is no MoE
leg for it to ride instead.

The two real-weights gates that stay local-only stay that way because of their
CHECKPOINTS, not their runtime:

- **MoE** — the smallest published `qwen3_5_moe` is 35B-A3B. Even the UD-Q2_K_XL quant
  peaks at 14.2 GB resident, past a standard macOS runner.
- **gemma4** — the only CI-reachable gemma4 is the gated
  `google/gemma-4-E2B-it-qat-mobile-transformers`, whose `sliding_window` is **512**.
  `gemma4_cold_tier_parity.rs` pins `SLIDING_WINDOW_TOKENS = 1024` and uses it as both
  the long gate's `min_restored_tokens` floor and the sub-window gate's ceiling
  assertion, so on a 512-window checkpoint those two assertions stop meaning what
  their messages say. Wiring it needs the window read from the loaded config first.

The structural fact both gemma4 gates rest on: with a `ColdSidecarPolicy` installed, a
non-zero `cached_tokens` in a _freshly loaded_ instance (empty hot cache) can only
have come from a validated sidecar. `gemma4_cold_tier_parity.rs` therefore runs **two**
gates, one per side of `min(boundary, window)`:

| gate                                            | prompt        | restored rotating state                        |
| ----------------------------------------------- | ------------- | ---------------------------------------------- |
| `gemma4_cold_tier_restart_parity`               | ~1.2k tokens  | post-wrap: a full window, `idx` inside the ring |
| `gemma4_cold_tier_restart_parity_sub_window`    | ~350 tokens   | pre-wrap: `boundary` rows                       |

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

Before `use_block_paged_cache` defaults can flip from `Some(false)` to enabled, four parity tests must pass on real weights:

| Test                                                        | Gate env var                    |
| ----------------------------------------------------------- | ------------------------------- |
| `crates/mlx-core/tests/qwen3_paged_vs_flat_parity.rs`       | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/lfm2_paged_vs_flat_parity.rs`        | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/gemma4_paged_vs_flat_parity.rs`      | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/qwen3_5_paged_vs_flat_parity.rs`     | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/qwen3_5_moe_paged_vs_flat_parity.rs` | `MLX_TEST_MODEL_PATH`           |
| `__test__/models/qwen3-paged-parity.test.ts`                | `QWEN3_PAGED_PARITY_MODEL_PATH` |

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
