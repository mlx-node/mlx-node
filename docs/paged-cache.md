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
down to the deepest such boundary a validated sidecar backs, or to zero. The boundary
is the `gdn_checkpoint_target` (the largest full block strictly before the end of the
prompt), and the recurrent state is only materialized there when the prefill **splits**
at that offset. So when a GDN cold policy is attached, `paged_prefill` forces that one
split even under the default single-shot chunk size
(`Qwen35Inner::cold_gdn_prefill_chunk_size`).

Be precise about what that split costs. It is **mathematically equivalent, not
bit-identical**: every attention query still attends over the whole cumulative range,
but the GDN scan runs as two launches instead of one, so the running state takes an
extra bf16 round trip at the boundary and the reduction order changes. That is the
same tradeoff vLLM mandates for `mamba_cache_mode == "align"`, which hard-requires
chunked prefill (`model_executor/models/config.py`) and is the regime Qwen3-Next runs
in — so it is the reference design rather than an invention. It does mean the split is
taken as soon as a policy is attached, i.e. on the FIRST persist-enabled run, before
anything has ever been restored. The restart-parity gate matched persist-on against
the persist-off baseline byte-for-byte on the gated checkpoints, which **bounds** the
divergence rather than proving it is absent at every prompt length.

Capture is text-only in v1 and never anchors deeper than the K/V chain actually
reached (`cold_captured_blocks`).

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

A gate on a large checkpoint is slow — the 26B gemma4 run takes ~66 min for the
long-prompt gate alone and the 35B MoE ~26 min, because a checkpoint with no
`.mlx-download-complete.json` marker full-shard-hashes its weights on every one of
the harness's fresh loads.

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
