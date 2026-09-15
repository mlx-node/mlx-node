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
assets into a small `.mlx-qwen4-assets-v2-*` directory beside the first split.
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

| Setting                      | Behavior                                                                |
| ---------------------------- | ----------------------------------------------------------------------- |
| `MLX_QWEN4_RESIDENCY=auto`   | Admit full or partial hot residency from current headroom.              |
| `MLX_QWEN4_RESIDENCY=full`   | Require the full hot set; fail if it cannot fit safely.                 |
| `MLX_QWEN4_RESIDENCY=stream` | Bound the on-demand cache to at most 8 GiB unless overridden.           |
| `MLX_QWEN4_WEIGHT_CACHE_GIB` | Optional integer ceiling, at least 2 GiB; live admission still applies. |
| `MLX_QWEN4_PREFILL_CHUNK`    | Power of two from 1–1024, constrained by the working-memory allowance.  |
| `MLX_QWEN4_PACKED_CACHE=0`   | Disable prepared SSD caching.                                           |
| `MLX_QWEN4_PACKED_CACHE_DIR` | Override the prepared-cache location.                                   |

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

## Validation and benchmarking

Small checked-in fixtures compare SafeTensors/GGUF and F32/BF16 outputs against
mlx-vlm, including sparse selection, paged continuation, MTP frontiers and media
positions. Tests also cover slot lifetimes, source corruption, bounded reads,
dynamic admission, cancellation, and specialized quantized kernels. Fixture
provenance and regeneration are in
[`tests/fixtures/qwen4-exp/README.md`](../crates/mlx-core/tests/fixtures/qwen4-exp/README.md).

The last pre-cleanup matched benchmark on an M5 Max with 128 GiB measured
**1,320.2 prefill tok/s and 21.39 decode tok/s** on a 1,024-token synthetic prompt
with 128 generated tokens, greedy AR, and no prefix reuse. Automatic weight
admission was 72 GiB / 454 expert slots; peak guarded footprint was 73.44 GiB.
Three measured samples followed two warmups per variant. The previous path
measured 1,292.8 prefill tok/s under the same protocol. All outputs matched.
These measurements describe that build and workload, not a 1,500-tok/s result
or a matched comparison with mlx.fast. A code-prompt comparison was too variable
to establish a reliable speedup. Compact evidence is in
[`qwen38-flash-next-performance.json`](research/qwen38-flash-next-performance.json).

Run the public-API benchmark against one built addon at a time. It records exact
prompt IDs, outputs, warmups, environment, memory admission and per-turn metrics.
Compare runs only with matching inputs, output lengths and residency, and keep
other models/compilers stopped. OS file cache, thermals and desktop load remain
uncontrolled.

```bash
MODEL_GUARD_RSS_GIB=88 MODEL_GUARD_OTHER_FOOTPRINT_GIB=8 \
MODEL_GUARD_REQUIRE_EVENT=passed QWEN4_BENCH_INPUT_TOKENS=1024 \
python3 scripts/guard-model-memory.py .cache/qwen4-benchmark.log \
  oxnode scripts/benchmark-qwen4.ts /models/first-split.gguf .cache/qwen4-benchmark.json
```

The guard monitors its process group's RSS, macOS physical footprint, memory
pressure and timeout, and stops only that group. Choose a ceiling appropriate to
the device and admitted budget; 88 GiB is the example for this 128-GiB machine.

For the optional full-checkpoint text, MTP, streaming, cancellation and image smoke:

```bash
MODEL_GUARD_RSS_GIB=48 MODEL_GUARD_REQUIRE_EVENT=passed \
MLX_QWEN4_WEIGHT_CACHE_GIB=32 MLX_QWEN4_PREFILL_CHUNK=512 \
python3 scripts/guard-model-memory.py .cache/qwen4-complete.log \
  oxnode scripts/test-qwen4-complete.ts /models/first-split.gguf /models/original-hf
```

Use `QWEN4_SMOKE_MODE=basic` for two text turns without auxiliary weights.
Full BF16 generation and real-model contexts beyond the 2,048-token sparse
threshold remain unvalidated; the small fixtures exercise sparse boundaries.
