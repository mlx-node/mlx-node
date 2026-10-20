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

The weight budget is calculated at bootstrap from actual
tensor sizes, physical RAM, and live free/file-backed memory. It reserves one
eighth of RAM for the system (bounded to 2–16 GiB), plus 4 GiB for runtime work,
and caps weights at 75% of physical RAM. Anonymous inactive and compressed
memory do not count as available. The budget is reconciled after tokenizer and
page-pool creation; growth rechecks headroom.

With a complete mapped expert pack (below), the load maps every routed expert
in place when the live host holds the whole pack beside what stays anonymous:
the pack's sections back the expert banks as no-copy Metal buffers, slot id
is expert id, and no expert is ever copied, missed or staged during a turn.
The weight budget then covers only what stays anonymous — fixed projection
banks and 2 GiB of row/staging room, through which auxiliary (MTP/vision)
rows stream like token rows — and the routed experts are clean page cache: the Metal
driver wires the banks a command buffer binds and releases them about a
second after the turn, so the process holds no expert memory between turns
and the OS may evict the pages. The decision is a byte budget for mapped
banks: physical RAM minus a 4 GiB margin, minus the host's live
non-reclaimable memory (wired, compressor, other processes' anonymous pages;
`vm_stat`, not "free"), minus that anonymous budget and the paged KV pool.
The pack maps when the budget covers every bank, else the slot path serves
the experts with its usual budget; `residencyInfo()` reports the budget
(`mappedBankBudgetBytes`), the live memory it saw (`liveNonReclaimableBytes`)
and the reason for a slot choice (`fallbackReason`). During a pass the driver
wires every bank, so a mapped turn cannot keep the slot path's free/file
reserve; it stops growing anonymous memory only below 2 GiB of free plus file
cache. A mapped turn that reads more than twice the pack from SSD, or runs
eight consecutive passes over ten times the running median of their kind
(a row count with no median yet — the decode steps of a process's first turn
— is judged against the fastest pass of 256 rows or more seen so far, and
such a pass counts only with a quarter GiB of SSD reads behind it: slower
without reads is GPU contention, recorded as the row count's own pace; the
streak and the SSD count start over when a turn opens on an idle model, whose
first pass is the bank re-wire; turns admitted while others run join the
window in progress), leaves the mapped path between that pass and its next
one: the same request goes on over slot banks with its KV and recurrent
state intact, every turn open with it (residency is one per model). Banks
and read-ahead are released first, then the slot budget is read from the live
host once wired memory has fallen and held for a second of fresh samples
(`host_statistics64` repeats its last answer between its few fresh reads per
second; such copies are skipped), at most 3 s, with the bootstrap rule over the
free plus file cache the host keeps once everything live is touched again
(free plus file plus purgeable, minus the inflation of the compressor pool and
the pages swapped out: the hog and the apps a wired pack pushed into the
compressor and into swap come back on top of the slots, whatever the plain
count says right after the unwire; 68 GiB of it beside a 40 GiB hog sized 48
GiB of slots that decoded at 0.11 tokens/s while the kernel cycled banks and
hog through the compressor); the prefill window is derived from the slot
allowance again and read before every remaining pass of the request; the
reason, the bytes or the step count and
the request's token position at the switch (`fallbackTokenIndex`) are
reported. Turns run without a wired-limit context (it does not pin
file-backed buffers and only delays their release), the mid-turn staging
admission is gone with the staging, the SSD safety limit subtracts the mapped
bytes MLX counts in `active_memory`, and a lookup window's decoded rows share
one Metal buffer (one per row made a fresh 10k-token prompt cost 165k Metal
resources and the third one abort at MLX's ~499k resource limit; rows a window
hits stay in their buffer, a buffer is charged whole until its last row is
evicted, and the LRU reports `decodedRowEvictions`). Greedy output no
longer depends on the free memory at load. The driver wires whole banks, not the experts a token
routes to, and unwires a bank about a second after its last use; a pass that
starts later than that after the previous one would otherwise re-wire the
layers one by one and fall into a 0.5 tokens/s cycle, so every main-network
forward pass whose predecessor started more than 0.9 s earlier first reads
one element of every mapped bank in a single eval (the MTP head's passes read
no mapped bank and neither touch nor count as a predecessor). The pack is
streamed into the page cache, never faulted in through the driver: when the
load has picked the mapped plan and read the dense banks, eight reader
threads start reading the whole pack in layer order with 4 MiB buffered
`pread`s, skipping every 4 MiB chunk that is already resident (one `mincore` over each file before its reads: between two reads of a file `mincore` reports the pages the kernel's read-ahead has in flight as in core, and a check per chunk skipped a third of an evicted pack unread, leaving it 54 of 73 GiB resident; a partly evicted pack costs only its holes, and every byte goes through a worker's own read with kernel read-ahead off (`F_RDAHEAD` 0; eight workers reach the SSD's rate without it); the stop and yield checks come before every chunk, resident or not, so nothing is read or in flight past the chunk a stop or a pending row allows, where an `F_RDADVISE` or the read-ahead would run on after the pool is dropped and ahead of the model's rows). Over the evicted pack this reaches 12.3 GiB/s against the SSD's
12.6, where one thread walking one file at a time behind its advisory
reached 9.3. The readers run at the default I/O tier: a throttled or
utility-tier read (`IOPOL_THROTTLE`, `IOPOL_UTILITY`) is paced behind other
readers, but its pages are speculative and 8 % of the pack survived such a
walk against a full page cache, so a lower tier would read the pack for
nothing. The cost is borne by whatever else reads the SSD during the ~6 s
stream (a cold 11 GiB read alongside it took 6.8 s instead of 0.95), except
the model's own row reads: while embedding or PLE rows are queued for the
reader threads or being read on the model thread, the workers wait between
two 4 MiB reads. One small read behind eight 4 MiB reads in the SSD queue
costs milliseconds, and a cold window's 2,000 inline row reads then held
layer 0 back until the stream was over (layer 0 began 6 s into a 7.1 s first
window). The load returns before the stream does; a turn that opens while it
runs changes nothing, a turn that opens over a pack that is not fully
resident with no stream running starts one. The
stream's order is every request's order, so a request never waits for a byte
it does not need, and a request that arrives after the stream finished finds
the pack resident. Resident is not wired: that request still pays the first
bind of every bank, the page-table work for 73 GiB of page-cache pages on the
model thread (about 2.5 s: a 6k prompt 8 s after load took 7.4 s against
4.7-5.2 warm and 11.4 cold), and the driver unwires a bank idle for about a
second, so nothing short of touching the banks while idle would make it
warm. While the pack streams, a stale pass does not touch: the
touch would bind every bank at once and fault the whole pack in on the model
thread before layer 0. Instead, every 700 ms, the banks of the layers already
run are touched so the driver keeps them wired through the long first window
(it unwires a bank idle for about 1.15 s; re-wiring all of them at the second
window cost about a second with the GPU idle). The reads go through the
files' descriptors, never through the banks' mapping (pages touched there
stay in the process's page tables for the mapping's life: 75 GiB of RSS
between turns) and never through the advisory alone (advisory pages are
speculative and were reclaimed before the driver bound them, doubling a cold
window's reads). The stream stops within one 4 MiB read when the store
leaves the mapped path or the model unloads; a cache reset does not stop it.
Nothing it reads is wired, the live-memory gate's inputs do not move (file
pages are reclaimable) and the tripwire's SSD count starts at the turn. A
prefill window also hands the reader threads the embedding and PLE rows of
the window after it, in file order, so the next window finds its rows ready
instead of reading them one `pread` at a time on the model thread.
Every prefill
window and every decode step needs the entire routed set resident at once:
when other processes' anonymous memory leaves less than that (a 20 GiB
working set beside the 78 GiB pack on 128 GiB), the kernel re-faults the pack
layer by layer and a 6k turn reads about 1 TB from SSD at 0.4 tokens/s
instead of 50; the live-memory gate and the tripwire above send such a host
to the slot path.

On the slot path, fixed projections are retained within a quarter of the
budget and routed experts use stable slots sized from the remaining budget and
tensor geometry. Sparse token/PLE rows stay on SSD: the 17 rows a token needs
(its embedding and 16 hashed PLE rows) are requested from four reader threads
as soon as the token is known, `pread` from descriptors kept open, and imported
lazily into the layer that consumes them, so the model thread neither opens
files nor drains the GPU at the token start; decoded rows are kept in the
weight LRU. Only selected cold experts are read. Every source read and its
F32-equivalent size are capped at 64 MiB; bank preparation accounts for both
imported chunks and their destination.

The routed expert tensors of a GGUF checkpoint are kept as a mapped expert
pack under `~/Library/Caches/mlx-node/qwen4-packed` on macOS:
`.mlx-qwen4-mapped-v3-<identity>/`, one file per `(layer, tensor)` whose three
quantized arrays are 16 KiB-aligned sections with the experts in id order
(~74 GiB for Flash Next). The first load builds it from the checkpoint,
hashing every section and reading each file back before its atomic rename;
later loads check the header and file size. Each tensor is built under its own
`.<tensor>.lock` file, which is never renamed or unlinked, so two loaders of
one checkpoint never rebuild or truncate each other's published file; a tensor
another process is building is left out of this load's pack. The build stops
when the disk falls under a 16 GiB reserve. Mapped residency maps the sections
directly; the slot path reads whole experts from them by range through the
page cache. A tensor the pack lacks, and every other matrix (fixed
projections, sparse lookup rows), is imported from the checkpoint. Deleting
the pack is safe while the model is stopped.

Each pack directory carries a `.manifest` recording its canonical source
directory and packed file list. Opening a checkpoint sweeps sibling
`.mlx-qwen4-mapped-v3-*` directories: one is orphaned only when its manifest
cannot prove liveness (missing or unreadable source) or records a different
identity than its directory name — and only once the newest
`.manifest`/`lock`/`.mapped` mtime is also older than one hour. The hour is a
grace gate for real orphan reasons, never an expiry on its own: a healthy
manifested pack stays cached indefinitely. Removal requires a non-blocking
lock on the pack's `lock` file, which every process serving or building the
pack holds shared; directories without a manifest are never removed, since
nothing can prove them stale, and a load never sweeps its own pack. Legacy
chunk directories named `.mlx-qwen4-packed-v2-<64 lowercase hex>`, which older
releases write, are removed automatically once idle for one hour and unlocked
(a non-blocking lock on their `budget` file), with or without a manifest; any
other name under that prefix is kept. The same
maintenance sweeps stale entries of the native GGUF cache under a 24 h grace
and per-entry locks; `.mlx-qwen4-assets-v2-*` tokenizer sidecars are exempt
from orphan removal and their markers are re-stamped on use, so they are only
collected by the `MLX_NATIVE_GGUF_CACHE_MAX_GB` least-recently-used cap.

A breached SSD-read safety limit reclaims the allocator freelist and re-polls
before erroring.

| Setting                        | Behavior                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLX_NATIVE_GGUF_CACHE_MAX_GB` | Optional cap on the whole native GGUF cache (packed weights plus extracted tokenizer assets), in GiB; least-recently-used eviction under a 24 h grace, with atomic root-wide admission at publish (a conversion that cannot fit loads from its staging dir instead; a later reload reuses that staged result rather than re-converting, and the refused dir is a per-pid load output the sweep keeps while the process lives). Unset, `<=0` or unparseable means unlimited. |

`residencyInfo()` reports the admitted bytes, hot inventory, resident banks,
expert-slot capacity (the expert count in mapped mode), the number of tensors
served from the mapped pack (`mappedPackFiles`), whether experts are mapped in
place (`mapped`, with `mappedBytes`; `weightBudgetBytes`,
`partialExpertSlots` and both follow a mid-session fallback), the bank budget
and live memory of the residency decision (`mappedBankBudgetBytes`,
`liveNonReclaimableBytes`), why the slot path serves the experts
(`fallbackReason`, with `fallbackReadBytes` or `fallbackSlowSteps`), the
prefill window, and the live decoded-row counters (`decodedRowLookups`,
`decodedRowMisses`, `decodedRowHitRate`, `decodedRowEvictions`,
`decodedRowEvictedBytes`, plus `prefetchedRows` / `lateRows`: imported rows
the reader threads had ready versus rows the model thread read or waited
for).
The budgets are runtime budgets; they do
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

The shared scheduler supports up to four live owners. QSA K/V pages are admitted
against a pool capped at 1 GiB; recurrent rows and MTP draft state are admitted
against the 4 GiB working allowance left after that pool and the prefill window
(a queued turn's reservation counts until its state exists, and a resumed
turn re-reserves its full token bound), and a turn is refused only when that
allowance cannot hold its state. The effective
context is bounded by cache geometry, the trained limit, and 32,768 tokens;
inspect `contextLimits()` for the admitted geometry. Multi-owner decode keeps stateful attention per owner and batches stateless
MLP projections across ready rows. Warm continuation reuses that
owner's full state; standalone KV-only prefix restores are disabled because
recurrent, PLE, and indexer state are also required.

Native MTP is **opt-in** (`enableMtp: true`); it drafts two tokens per cycle
unless `mtpDepth` says otherwise, capped at three. It runs
only on the shared scheduler. An ownerless `enableMtp` start or continuation runs
there as the ownerless session that ownerless AR turns continue; one that still
reaches the exclusive lane (`MLX_SERVE_FORCE_SERIAL`) is refused before prefill
instead of decoding plain AR. Proposals use private draft state; verification
runs every draft row through each layer together, reading the layer's weights
once; over the mapped pack its routes stay on the device and one readback at
the end checks every layer, over slot banks each layer reads its routes back
to fill its slots. It commits only accepted target
frontiers, whose head rows are written by the next proposal's first draft pass
together with its anchor row. The head's routed experts become resident Q8
(affine, group 32) banks on the first MTP owner, about 2.6 GiB for the released
checkpoint outside the weight LRU, and each draft step gathers them on the
device. Both banks are installed together, only when the weight cache can stage
them; otherwise drafts keep streaming expert rows, and
`residencyInfo().residentBankBytes` reports the banks once they exist.
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
