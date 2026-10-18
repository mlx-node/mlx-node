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

With a complete mapped expert pack (below) and a hot set within 75% of
physical RAM, `auto` maps every routed expert in place: the pack's sections
back the expert banks as no-copy Metal buffers, slot id is expert id, and no
expert is ever copied, missed or staged during a turn. The weight budget then
covers only what stays anonymous — fixed projection banks, up to 8 GiB of
auxiliary (MTP/vision) inventory and 2 GiB of row/staging room — and the
routed experts are clean page cache: the Metal driver wires the banks a
command buffer binds and releases them about a second after the turn, so the
process holds no expert memory between turns and the OS may evict the pages.
Turns run without a wired-limit context (it does not pin file-backed buffers
and only delays their release), the mid-turn staging admission is gone with
the staging, the forward and growth headroom checks still guard what stays
anonymous (rows, auxiliary weights, fallback projection chunks, scratch), and
the SSD safety limit subtracts the mapped bytes MLX counts in `active_memory`. Greedy output no longer depends on the free
memory at load. The driver wires whole banks, not the experts a token
routes to, and unwires a bank about a second after its last use; a pass that
starts later than that after the previous one would otherwise re-wire the
layers one by one and fall into a 0.5 tokens/s cycle, so every main-network
forward pass whose predecessor started more than 0.9 s earlier first reads
one element of every mapped bank in a single eval (the MTP head's passes read
no mapped bank and neither touch nor count as a predecessor). When such a
pass finds the pack not fully resident (eight 256 KiB `mincore` probes per
file), it reads ahead instead of touching: a worker thread asks the kernel
(`F_RDADVISE`, which skips resident pages) for the whole files of layers 0
to 3 as the pass starts and for layer L+3 as layer L starts, so a cold first
prompt streams the pack at the SSD's rate (10.5 GiB/s measured) rather than
the driver's fault rate (4.2 GiB/s), and the model thread never reads; the
touch would bind every bank at once and fault the whole pack in before
layer 0. The worker reads one byte of every page it requested (advisory
pages are speculative and were reclaimed before the driver bound them,
doubling a cold window's reads), so after a cold turn the pack counts in
the process RSS while its footprint is unchanged. `MLX_QWEN4_PREFETCH_LAYERS`
sets the depth (default 3; `0` restores the unconditional touch) and
`MLX_QWEN4_PREFETCH_AT_LOAD=1` requests the whole pack as the model loads.
Every prefill
window and every decode step needs the entire routed set resident at once:
when other processes' anonymous memory leaves less than that (a 20 GiB
working set beside the 78 GiB pack on 128 GiB), the kernel re-faults the pack
layer by layer and a 6k turn reads about 1 TB from SSD at 0.4 tokens/s
instead of 50. The 75% gate does not see that memory;
`MLX_QWEN4_MAPPED_PACK=0` keeps the slot path for A/B and for such hosts;
`MLX_QWEN4_WEIGHT_CACHE_GIB`, `stream` and `full` also select it.

Otherwise, if the hot text network fits, it is prepared once in resident
quantized banks; else fixed projections are retained within a quarter of the
budget and routed experts use stable slots sized from the remaining budget and
tensor geometry. Sparse token/PLE rows stay on SSD. Only selected cold experts
are read. Every source read and its F32-equivalent size are capped at 64 MiB;
bank preparation accounts for both imported chunks and their destination.

The optional prepared SSD cache stores losslessly imported matrix chunks, so
later RAM evictions or restarts do not repeat the import. It excludes sparse
lookup rows, verifies checksums before importing arrays, and publishes files
atomically. Each checkpoint/cache version is capped at 96 GiB and leaves at
least 16 GiB disk space. It defaults to `~/Library/Caches/mlx-node/qwen4-packed`
on macOS. Deleting that cache is safe while the model is stopped.

Each identity directory also carries a `.manifest` recording its canonical
source directory and packed file list. Opening the cache sweeps sibling
`.mlx-qwen4-packed-v2-*` and `.mlx-qwen4-mapped-v3-*` directories: one is
orphaned only when its manifest cannot prove liveness (missing or unreadable
source) or records a different identity than its directory name — and only
once the newest `budget`/`.manifest`/`.pack`/`.mapped` mtime is also older
than one hour. The hour is a grace gate for real orphan reasons, never an
expiry on its own: a healthy manifested dir stays cached indefinitely until
`MLX_QWEN4_PACKED_CACHE_MAX_GB` evicts it. Removal requires a non-blocking
lock on that directory's `budget` ledger — a live writer holds it — or, for a
mapped pack, on its `lock` file, which every process serving or building the
pack holds shared; directories without a manifest are never removed, since
nothing can prove them stale. Under the ledger lock, its `used` figure is
reconciled against the `.pack` bytes actually on disk. The two directories of
one identity are one checkpoint's cache: a load never sweeps or evicts its
own chunk dir or mapped dir. The same maintenance sweeps stale entries of the native
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
| `MLX_QWEN4_WEIGHT_CACHE_GIB` | Optional integer ceiling, at least 2 GiB; live admission still applies. Pins an anonymous budget, so the slot path is used even when the mapped pack exists. |
| `MLX_QWEN4_MAPPED_PACK=0`    | Keep the slot path although the mapped pack is complete (A/B arm).      |
| `MLX_QWEN4_PREFETCH_LAYERS`  | Layers of the mapped pack read ahead of the one computing when a pass starts over a pack that is not fully resident (default 3; `0` disables read-ahead and a stale pass always touches every bank). |
| `MLX_QWEN4_PREFETCH_AT_LOAD=1` | Ask the kernel for the whole mapped pack as the model loads (opt-in; moves the first prompt's SSD time into the load). |
| `MLX_QWEN4_PREFILL_CHUNK`    | Power of two from 1–1024, constrained by the working-memory allowance.  |
| `MLX_QWEN4_BATCHED_QSA=0`    | Run prompt windows past the indexer budget one token at a time instead of the batched sparse-attention kernel. |
| `MLX_QWEN4_PACKED_CACHE=0`   | Disable prepared SSD caching.                                           |
| `MLX_QWEN4_PACKED_CACHE_DIR` | Override the prepared-cache location.                                   |
| `MLX_QWEN4_PACKED_CACHE_MAX_GB` | Optional global cap across every packed-cache identity directory, in GiB; least-recently-used eviction. Unset or `<=0` keeps only the per-checkpoint 96 GiB budget. |
| `MLX_NATIVE_GGUF_CACHE_MAX_GB` | Optional cap on the whole native GGUF cache (packed weights plus extracted tokenizer assets), in GiB; least-recently-used eviction under a 24 h grace, with atomic root-wide admission at publish (a conversion that cannot fit loads from its staging dir instead; a later reload reuses that staged result rather than re-converting, and the refused dir is a per-pid load output the sweep keeps while the process lives). Unset, `<=0` or unparseable means unlimited. |
| `MLX_QWEN4_IGNORE_MEMORY_PRESSURE` | `=1` skips the macOS bootstrap memory-pressure gate — diagnostic only; the free/file-backed headroom checks still apply. |

The routed expert tensors of a GGUF checkpoint are also kept as a mapped expert
pack beside the chunk cache: `.mlx-qwen4-mapped-v3-<identity>/`, one file per
`(layer, tensor)` whose three quantized arrays are 16 KiB-aligned sections with
the experts in id order (~74 GiB for Flash Next). The first load builds it from
the verified chunks (else from the checkpoint), hashing every section and
reading each file back before its atomic rename; later loads check the header
and file size. Each tensor is built under its own `.<tensor>.lock` file, which
is never renamed or unlinked, so two loaders of one checkpoint never rebuild
or truncate each other's published file; a tensor another process is building
is left out of this load's pack. Mapped residency maps its sections
directly; the slot path reads whole experts from it by range through the page
cache instead of opening, locking, stamping and hashing one chunk file per
expert; a tensor the pack lacks (the build stops when the
disk falls under the 16 GiB reserve, or when `MLX_QWEN4_PACKED_CACHE_MAX_GB`
leaves no room after evicting idle siblings) keeps the chunk reader. The
mapped dir's `.mapped` and staging bytes count toward that cap like `.pack`
bytes — a file is admitted and its staging file sized to its final length
under one root-wide lock, so concurrent builders never admit against the same
room — and an orphaned or least-recently-used mapped dir is removed by the
same sweep — except one whose `lock` file a process holds shared, which every
open or building pack does.

`residencyInfo()` reports the admitted bytes, hot inventory, resident banks,
expert-slot capacity (the expert count in mapped mode), the number of tensors
served from the mapped pack (`mappedPackFiles`), whether experts are mapped in
place (`mapped`, with `mappedBytes`), and prefill window. These are runtime budgets; they do
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
