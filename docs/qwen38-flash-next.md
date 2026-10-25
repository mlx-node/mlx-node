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
GGUF asset root. `MLX_NATIVE_GGUF_CACHE_DIR` overrides that asset root;
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

## Memory and SSD residency

The weight budget is calculated at bootstrap from actual tensor sizes,
physical RAM, and live free/file-backed memory. It reserves one eighth of RAM
for the system (bounded to 2–16 GiB), plus 4 GiB for runtime work, and caps
weights at 75% of physical RAM. Anonymous inactive and compressed memory do
not count as available. The budget is reconciled after tokenizer and page-pool
creation; growth rechecks headroom.

For a supported GGUF checkpoint, the load first offers the routed experts full
direct residency: anonymous banks at the checkpoint's full expert count, with
slot id equal to expert id. The live-memory gate must admit the whole bank set
beside fixed projection banks, sparse-row/staging room, the paged KV pool, and
any MTP draft banks. Its bank allowance is physical RAM minus a 4 GiB margin,
the host's live non-reclaimable memory (wired, compressor, and anonymous pages
other than purgeable pages), and that other anonymous need. Full direct banks
have no misses or LRU replacement.

Eight reader threads fill every layer's three banks from the main GGUF shard
set through the GPU repack pipeline, in layer order, with at most 8 MiB per
read window and four staging buffers. A cold layer-major prefill waits only for
the layer it is about to consume. Canonically ordered split GGUFs use the same
path, including a metadata-only first shard: aligned read windows may cross
shard boundaries, and every shard's identity and tensor bounds are checked
before publication. A changed, missing, or truncated source fails the fill;
partially written banks are never published.

Below the full-bank gate, the remaining bank budget sizes and caps a partial
slot capacity. Partial misses use the same direct-GGUF GPU fill pipeline and
8 MiB read bound. Prefill keeps its last-assignment expert groups, whose expert
sets are disjoint, so a cold window reads each selected expert at most once per
layer. `direct` reports only the full identity-resident plan; partial direct
fills report `direct: false`. Unsupported source layouts retain the bounded
per-row CPU checkpoint importer; they do not select another residency format.

Between turns, eligible Metal bank buffers become purgeable. A buffer found
empty at the next turn is refilled from GGUF before its layer is used. The
30-second idle fallback explicitly empties banks, and tests cover both that
path and purge/refill equality. A full fill from a slow source keeps its banks
resident at idle instead of repeatedly paying the same source latency.

Fixed projections are retained within their admitted allowance. Sparse
token/PLE rows stay on SSD: the 17 rows a token needs (its embedding and 16
hashed PLE rows) are requested from 32 reader threads as soon as the token is
known, read through descriptors kept open, and imported lazily into the layer
that consumes them. Decoded rows are kept in the weight LRU. Every source read
and its F32-equivalent size are capped at 64 MiB; bank preparation accounts for
both imported chunks and their destination. A breached SSD-read safety limit
reclaims the allocator freelist and re-polls before erroring.

Flash Next does not build or open a derived weight pack. A load creates no
files or directories under the retired `qwen4-packed` cache root. The small
`.mlx-qwen4-assets-v2-*` config/tokenizer sidecar described above is separate
and remains in use.

For one release, opening a GGUF checkpoint also performs best-effort cleanup of
retired packs under `~/Library/Caches/mlx-node/qwen4-packed` on macOS
(`~/.cache/mlx-node/qwen4-packed` elsewhere). Only real directories named
`.mlx-qwen4-mapped-v3-<64 lowercase hex>` or
`.mlx-qwen4-packed-v2-<64 lowercase hex>` are eligible. They must have been
idle for one hour and yield to a nonblocking exclusive lock on the legacy
`lock` or `budget` file, respectively. Fresh directories, legacy readers that
still hold a lock, symlinks, ordinary files, foreign suffixes, and other cache
entries are retained. The sweep creates no root, marker, or replacement pack.

### Partial-capacity route replay

The test-only replay reads schema v1 `qwen4_exp_expert_route_trace` JSONL with
explicit line/schema validation, applies the production last-assignment prefill
grouping and per-layer LRU rules, and reports target and prefill-plus-target
miss bytes per output token. It accepts an optional `lru-results.json` from the
P3.0-c `replay_lru.py` tool for field-by-field comparison.

The P3.0-c real capture was blocked by host memory and produced **no valid
real-model traces**. No real capacity table or 40 GiB-hog throughput result is
claimed here. The checked replay expectations use clearly labeled synthetic
traces; fixture inference separately checks two capacities, bounded banks,
completion, refill, and source errors without representing itself as a real
model benchmark. A future valid capture can be replayed with the ignored
`real_trace_partial_replay` test and the variables below.

| Setting                    | Behavior                                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `QWEN4_EXP_ROUTE_TRACE`   | A valid P3.0-c JSONL route trace for the ignored partial replay test.                                                                   |
| `QWEN4_EXP_REPLAY_CAPACITIES` | Comma-separated replay capacities (default `128,192,256,384,512`).                                                                  |
| `QWEN4_EXP_REPLAY_SLOT_COST` | Bytes charged per unique missing `(layer, expert)` (default 8 MiB).                                                                    |
| `QWEN4_EXP_REPLAY_EXPECTED` | Optional P3.0-c `lru-results.json`; every replay field must match.                                                                     |
| `QWEN4_EXP_REPLAY_LABEL`  | Capture label selecting one expected row when the results contain multiple captures.                                                    |
| `QWEN4_EXP_REAL_MODEL`     | Opt-in real-model tests (ignored by default): the Qwen3.8 GGUF path.                                                                  |
| `QWEN4_EXP_REAL_INPUTS`   | Their prompts: a `{name, tokenIds}` case list from `oxnode scripts/benchmark-fixture.ts fetch --fixture qwen38 --output inputs.json`. |
| `QWEN4_EXP_REAL_AUX`      | The MTP checkpoint dir the slot tests attach.                                                                                         |
| `QWEN4_EXP_REAL_STEPS`    | Greedy decode steps per case (default 256).                                                                                           |
| `QWEN4_EXP_REAL_CASES`    | Comma-separated case names (default `6k,16k`).                                                                                        |
| `QWEN4_EXP_REAL_EXPECTED` | A `QWEN4_EXP_REAL_OUT` file of an earlier run: each case's greedy argmax must equal its `reference.argmax`.                           |
| `QWEN4_EXP_REAL_OUT`      | Where the greedy test writes each case's argmax and tokens.                                                                           |

Run them with `QWEN4_EXP_REAL_MODEL=... QWEN4_EXP_REAL_INPUTS=... cargo test -p mlx-core --lib real_ -- --ignored --test-threads=1 --nocapture`.

`residencyInfo()` reports the admitted bytes, hot inventory, resident banks,
partial-slot capacity (the expert count at full direct residency), `direct`
and `directBytes`, and layered fill progress (`directLayers`,
`directFilledLayers`, `directFillInProgress`). The bank allowance and live
memory of the residency decision are `bankBudgetBytes` and
`liveNonReclaimableBytes`; `fallbackReason` is present when that gate selected
partial slots. The prefill window and live decoded-row counters are
`decodedRowLookups`, `decodedRowMisses`, `decodedRowHitRate`,
`decodedRowEvictions`, `decodedRowEvictedBytes`, plus `prefetchedRows` /
`lateRows` (imports the reader threads had ready versus rows the model thread
read or waited for).
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

The PLE key and value projections run an admitted Q8 bank (affine, group 32,
BF16 input) through the row-exact decode kernel, four vectors per call: a
prefill window of T tokens makes ceil(T / 4) calls per weight chunk instead of
one gather GEMV per token, and MTP verification projects all of its rows' PLE
embeddings in one call before the gate and convolution advance row by row.
Every row keeps the bits of its single-row projection.

The decode mixer activation (the hyper-connection down projection, Q8
320 x 10240, with its SiLU epilogue) splits K across the eight simdgroups of a
threadgroup per output: each walks five of the forty 256-column blocks and
stores every block's term, and simdgroup 0 adds the forty terms in ascending
block order, the order of the one-simdgroup chain it replaces, before the same
`simd_sum`. Every F32 sum and BF16 output keeps its bits; a partial sum per
simdgroup would reassociate the chain and is not used.

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
once. Full direct banks keep routes on the device with one readback at the
end; partial banks read routes back to fill their slots. It commits only
accepted target frontiers, whose head rows are written by the next proposal's
first draft pass together with its anchor row. The head's routed experts become
resident Q8 (affine, group 32) banks at load whenever the head is attached,
before the direct expert fill starts, about 2.83 GiB for the released checkpoint outside the weight
LRU, and each draft step gathers them on the device. Both banks are installed
together, only when the weight cache can stage them; otherwise the load goes on
and drafts keep streaming expert rows for the life of the process.
`residencyInfo().residentBankBytes` includes the banks from load on.
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
