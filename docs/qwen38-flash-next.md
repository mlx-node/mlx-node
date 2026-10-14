# Qwen3.8-Flash-Next

The `qwen4_exp` runtime supports Qwen3.8-Flash-Next text generation, streaming,
native MTP, image input, and paged request scheduling. `loadModel()` and
`loadSession()` detect Hugging Face `qwen4_exp` / `qwen4_exp_text` configs and
GGUF's `qwen4exp` architecture.

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('/models/Qwen3.8-Flash-Next');
const response = await session.send('What is the capital of France?', {
  config: { temperature: 0, maxNewTokens: 16, reasoningEffort: 'none' },
});
console.log(response.text);
```

## Checkpoints

A Hugging Face directory needs its config, tokenizer assets, and all referenced
SafeTensors shards. Original BF16 weights are read in bounded chunks; they are
never loaded as one dense model.

For [Unsloth UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q4_K_XL),
place all four splits together and pass the first `.gguf` file. The loader
validates the split descriptors before inference. It extracts embedded tokenizer
assets into a small `.mlx-qwen4-assets-v2-*` directory in the writable native
GGUF application cache. `MLX_NATIVE_GGUF_CACHE_DIR` overrides that cache;
otherwise the loader uses the XDG/OS user cache, with a temporary-directory
fallback. The source checkpoint directory only needs read access.
Quantized weight codes and scale/minimum values are preserved during import;
loading does not require a whole-checkpoint conversion or requantization.

```typescript
const session = await loadSession('/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf', {
  auxiliaryModelPath: '/models/Qwen3.8-Flash-Next',
});
```

`auxiliaryModelPath` supplies the matching original checkpoint's MTP and vision
weights when GGUF omits them. Geometry is checked before attachment, and those
weights load only when used. Text-only GGUF generation needs no auxiliary model.

## Memory and SSD caching

The default `auto` policy calculates the weight budget at bootstrap from actual
tensor sizes, physical RAM, and live free/file-backed memory. It reserves one
eighth of RAM for the system (bounded to 2–16 GiB), plus 4 GiB for runtime work,
and caps weights at 75% of physical RAM. Anonymous inactive and compressed
memory do not count as available. The budget is reconciled after tokenizer and
page-pool creation; growth rechecks headroom and memory pressure.

If the hot text network fits, it is prepared once in resident quantized banks.
Otherwise, fixed projections are retained within a quarter of the budget and
routed experts use stable slots sized from the remaining budget and tensor
geometry. Sparse token/PLE rows stay on SSD. Only selected cold experts are
read. Every source read and its F32-equivalent size are capped at 64 MiB;
bank preparation accounts for both imported chunks and their destination.

The optional prepared SSD cache stores losslessly imported matrix chunks, so
later RAM evictions or restarts do not repeat the import. It excludes sparse
lookup rows, verifies checksums before importing arrays, and publishes files
atomically. Each checkpoint/cache version is capped at 96 GiB and leaves at
least 16 GiB disk space. It defaults to `~/Library/Caches/mlx-node/qwen4-packed`
on macOS. Deleting that cache is safe while the model is stopped.

Each identity directory also carries a `.manifest` recording its canonical
source directory and packed file list. Opening the cache sweeps sibling
`.mlx-qwen4-packed-v2-*` directories: one is orphaned only when its manifest
cannot prove liveness (missing or unreadable source) or records a different
identity than its directory name — and only once the newest
`budget`/`.manifest`/`.pack` mtime is also older than one hour. The hour is a
grace gate for real orphan reasons, never an expiry on its own: a healthy
manifested dir stays cached indefinitely until `MLX_QWEN4_PACKED_CACHE_MAX_GB`
evicts it. Removal requires a non-blocking lock on that
directory's `budget` ledger — a live writer holds it — and directories without
a manifest are never removed, since nothing can prove them stale. Under the
same lock, the ledger's `used` figure is reconciled against the `.pack` bytes
actually on disk. The same maintenance sweeps stale entries of the native
GGUF cache under a 24 h grace and per-entry locks; `.mlx-qwen4-assets-v2-*`
tokenizer sidecars are exempt from orphan removal and their markers are
re-stamped on use, so they are only collected by the
`MLX_NATIVE_GGUF_CACHE_MAX_GB` least-recently-used cap.

Bootstrap, admission and the SSD-read safety limit also gate on macOS memory
pressure rather than failing on the first reading: pressure is polled every
250 ms for up to about 3 s before the load is refused, and a breached safety
limit reclaims the allocator freelist and re-polls before erroring.
`MLX_QWEN4_IGNORE_MEMORY_PRESSURE=1` bypasses the gate entirely.

| Setting                      | Behavior                                                                |
| ---------------------------- | ----------------------------------------------------------------------- |
| `MLX_QWEN4_RESIDENCY=auto`   | Admit full or partial hot residency from current headroom.              |
| `MLX_QWEN4_RESIDENCY=full`   | Require the full hot set; fail if it cannot fit safely.                 |
| `MLX_QWEN4_RESIDENCY=stream` | Bound the on-demand cache to at most 8 GiB unless overridden.           |
| `MLX_QWEN4_WEIGHT_CACHE_GIB` | Optional integer ceiling, at least 2 GiB; live admission still applies. |
| `MLX_QWEN4_PREFILL_CHUNK`    | Power of two from 1–1024, constrained by the working-memory allowance.  |
| `MLX_QWEN4_PACKED_CACHE=0`   | Disable prepared SSD caching.                                           |
| `MLX_QWEN4_PACKED_CACHE_DIR` | Override the prepared-cache location.                                   |
| `MLX_QWEN4_PACKED_CACHE_MAX_GB` | Optional global cap across every packed-cache identity directory, in GiB; least-recently-used eviction. Unset or `<=0` keeps only the per-checkpoint 96 GiB budget. |
| `MLX_NATIVE_GGUF_CACHE_MAX_GB` | Optional cap on the whole native GGUF cache (packed weights plus extracted tokenizer assets), in GiB; least-recently-used eviction under a 24 h grace, with atomic root-wide admission at publish (a conversion that cannot fit loads from its staging dir instead). Unset, `<=0` or unparseable means unlimited. |
| `MLX_QWEN4_IGNORE_MEMORY_PRESSURE` | `=1` skips the macOS bootstrap memory-pressure gate — diagnostic only; the free/file-backed headroom checks still apply. |

`residencyInfo()` reports the admitted bytes, hot inventory, resident banks,
expert-slot capacity, and prefill window. These are runtime budgets; they do
not include every allocation made by macOS or other applications. The runtime
currently requires macOS Metal for paged inference. NAX kernels are selected
only for supported hardware and shapes; other shapes retain the native fallback.

## Execution and limits

Prefill batches projections across bounded windows. Eligible NAX expert kernels
read original token rows indirectly, fuse gate/up work, and keep weights packed.
Batched rotary tables and compact dense projections reduce intermediate arrays.
Paged K/V writes accept read-only contiguous offset views while preserving strict
mutable-pool validation. GDN recurrent state remains F32, and BF16 rounding
boundaries are retained where required for parity.

The shared scheduler supports up to four live owners, subject to attention-page
and recurrent-state admission. QSA K/V uses a pool capped at 1 GiB. The effective
context is bounded by cache geometry, the trained limit, and 32,768 tokens;
inspect `contextLimits()` for the admitted geometry. Multi-owner decode keeps stateful attention per owner and batches stateless
MLP projections across ready rows. Warm continuation reuses that
owner's full state; standalone KV-only prefix restores are disabled because
recurrent, PLE, and indexer state are also required.

Native MTP is **opt-in** (`enableMtp: true`) with depth capped at three. Proposals
use private draft state; verification commits only accepted target frontiers.
Head availability alone does not establish a latency benefit. Stochastic MTP
keeps distribution-correct verification and disables the greedy adaptive policy.

Image turns run behind an exclusive scheduler barrier. Up to four images are
accepted per rendered conversation, with a 16-megapixel / 32-MiB input limit per
image and bounded preprocessing. Use `releaseCacheOwner()` or `resetCaches()` to
release idle owners. Audio, video, and image-bearing MTP are unsupported.

## Validation and performance

Small checked-in fixtures compare SafeTensors/GGUF and F32/BF16 outputs against
mlx-vlm, including sparse selection, paged continuation, MTP frontiers and media
positions. Tests also cover slot lifetimes, source corruption, bounded reads,
dynamic admission, cancellation, and specialized quantized kernels. See the
[fixture provenance](../crates/mlx-core/tests/fixtures/qwen4-exp/README.md).

The [consolidated research report](research/qwen38-flash-next.md) records the
mlx.fast source ports, exact and experimental arithmetic, workload/caching
conditions, correctness checks, and measurements. Historical macOS 26 runs
exceeded 90% of the published reference prefill rate with experimental arithmetic
enabled; the current macOS 27 recheck does not establish that target, and decode
remains below it. The checkpoints and prompts differ from the private reference.

Full BF16 generation and real-model contexts beyond the 2,048-token sparse
threshold remain unvalidated; the small fixtures exercise sparse boundaries.
